#!/usr/bin/env python3
"""
EXOSFEAR MicroGo KG v4 — Regime Nets + Dreaming

5-expert graph team: opening, tactics, territory, endgame, MEMORY
Memory expert is backed by a population of self-naming Regime Nets that
recognize board situations (e.g. "NW_fight_early", "center_pressure_late").

Short-term KG buffer records raw experience. During DREAMING, regimes are
born from experience clusters, trained, merged, pruned, and auto-named.

    pip install flask torch numpy    # full
    pip install flask numpy          # demo
    python microgo.py                # menu
"""
from __future__ import annotations
import argparse,base64,concurrent.futures,gzip,hashlib,io,ipaddress,json
import math,os,pickle,random,secrets,signal,socket,subprocess,sys
import threading,time
from collections import defaultdict,deque
from dataclasses import dataclass,field
from typing import Any,Dict,List,Optional,Tuple
import numpy as np
MOCK_MODE=False
try:
    import torch,torch.nn as nn,torch.nn.functional as F
    torch.set_num_threads(1)
    try:torch.set_num_interop_threads(1)
    except:pass
except:MOCK_MODE=True
BOARD_SIZE=6;PASS_MOVE=36;ALL_MOVES=37;KOMI=3.5;MAX_GAME_LEN=120;SEED=42
EXPERT_NAMES=["opening","tactics","territory","endgame","memory"]
VERSION="exosfear-microgo-kg-4.0";DEFAULT_COORD_PORT=5000;DEFAULT_WORKER_PORT=8765
def now_ts():return time.strftime("%Y-%m-%d %H:%M:%S")
def choose_device():return "cuda" if not MOCK_MODE and torch.cuda.is_available() else "cpu"

# === NET HELPERS ===
def get_local_ips():
    ips=[]
    def add(ip):
        ip=(ip or"").strip()
        if ip and not ip.startswith("127.") and":"not in ip and ip not in ips:ips.append(ip)
    try:
        for res in socket.getaddrinfo(socket.gethostname(),None,socket.AF_INET):add(res[4][0])
    except:pass
    for p in["8.8.8.8","1.1.1.1","192.168.1.1"]:
        try:s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);s.connect((p,80));add(s.getsockname()[0]);s.close()
        except:pass
    return ips
def port_is_free(h,p):
    try:
        with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as s:s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);s.bind((h,p))
        return True
    except:return False
def find_free_port(h="0.0.0.0",pref=8765):
    for p in range(pref,pref+200):
        if port_is_free(h,p):return p
    return 0
def resolve_port(pref,label=""):
    if port_is_free("0.0.0.0",pref):return pref
    alt=find_free_port("0.0.0.0",pref+1)
    try:
        pid=None
        try:pid=int(subprocess.check_output(["lsof","-ti",str(pref)],stderr=subprocess.DEVNULL).decode().strip().split()[0])
        except:pass
        print(f"\n  Port {pref} in use."+(" PID:"+str(pid) if pid else ""))
        ch=input(f"  [K]ill / [U]se {alt}? [K]: ").strip().lower() or"k"
        if ch=="k"and pid:os.kill(pid,signal.SIGTERM);time.sleep(0.5);return pref if port_is_free("0.0.0.0",pref) else alt
    except:pass
    return alt
def http_json(url,method="GET",payload=None,token="",timeout=300):
    import urllib.request;data=json.dumps(payload).encode() if payload is not None else None
    req=urllib.request.Request(url=url,data=data,method=method);req.add_header("Content-Type","application/json")
    if token:req.add_header("X-Token",token)
    with urllib.request.urlopen(req,timeout=timeout) as r:return json.loads(r.read().decode())
def scan_subnet(base_ip,port,token=""):
    found=[]
    try:net=ipaddress.ip_network(base_ip+"/24",strict=False)
    except:return found
    def ck(ip):
        try:
            d=http_json(f"http://{ip}:{port}/health",token=token,timeout=1)
            if d.get("ok")and d.get("role")=="microgo_worker":d["url"]=f"http://{ip}:{port}";return d
        except:pass
    with concurrent.futures.ThreadPoolExecutor(48) as ex:
        for f in concurrent.futures.as_completed({ex.submit(ck,str(ip)):ip for ip in net.hosts()}):
            try:
                r=f.result()
                if r:found.append(r)
            except:pass
    return sorted(found,key=lambda x:x.get("url",""))
def ping_workers(urls,token):
    out=[]
    for u in urls:
        try:d=http_json(u.rstrip("/")+"/health",token=token,timeout=5);out.append({"url":u,**d})
        except Exception as e:out.append({"url":u,"ok":False,"error":str(e)[:80]})
    return out
def compress_obj(o):return base64.b64encode(gzip.compress(pickle.dumps(o,protocol=pickle.HIGHEST_PROTOCOL))).decode("ascii")
def decompress_obj(s):return pickle.loads(gzip.decompress(base64.b64decode(s.encode("ascii"))))
def move_to_str(m):return"pass"if m==PASS_MOVE else f"{chr(65+m%BOARD_SIZE)}{m//BOARD_SIZE+1}"

# === GO ENGINE ===
class GoState:
    __slots__=("board_data","to_play","passes","move_count","_history")
    def __init__(s,board=None,to_play=1,passes=0,move_count=0,history=None):
        s.board_data=board if board is not None else np.zeros((BOARD_SIZE,BOARD_SIZE),dtype=np.int8)
        s.to_play=to_play;s.passes=passes;s.move_count=move_count;s._history=history if history is not None else frozenset()
    @staticmethod
    def new():s=GoState();s._history=frozenset([s.board_data.tobytes()]);return s
    def board_array(s):return s.board_data.copy()
    def _nbrs(s,r,c):
        o=[];r>0 and o.append((r-1,c));r+1<BOARD_SIZE and o.append((r+1,c));c>0 and o.append((r,c-1));c+1<BOARD_SIZE and o.append((r,c+1));return o
    def _gl(s,b,r,c):
        color=int(b[r,c]);stk=[(r,c)];seen={(r,c)};grp=[];libs=set()
        while stk:
            rr,cc=stk.pop();grp.append((rr,cc))
            for nr,nc in s._nbrs(rr,cc):
                v=int(b[nr,nc])
                if v==0:libs.add((nr,nc))
                elif v==color and(nr,nc)not in seen:seen.add((nr,nc));stk.append((nr,nc))
        return grp,libs
    def legal_moves(s):
        mvs=[];b=s.board_array()
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if b[r,c]==0 and s.try_play(r*BOARD_SIZE+c)is not None:mvs.append(r*BOARD_SIZE+c)
        mvs.append(PASS_MOVE);return mvs
    def try_play(s,mv):
        b=s.board_array()
        if mv==PASS_MOVE:return GoState(b,-s.to_play,s.passes+1,s.move_count+1,s._history)
        r,c=divmod(mv,BOARD_SIZE)
        if b[r,c]!=0:return None
        b[r,c]=s.to_play
        for nr,nc in s._nbrs(r,c):
            if b[nr,nc]==-s.to_play:
                g,l=s._gl(b,nr,nc)
                if not l:
                    for gr,gc in g:b[gr,gc]=0
        g,l=s._gl(b,r,c)
        if not l:return None
        bh=b.tobytes()
        if bh in s._history:return None
        return GoState(b,-s.to_play,0,s.move_count+1,s._history|{bh})
    def game_over(s):return s.passes>=2 or s.move_count>=MAX_GAME_LEN
    def final_score_black(s):
        b=s.board_array();sb=int(np.sum(b==1));sw=int(np.sum(b==-1));vis=np.zeros_like(b,dtype=np.uint8);tb=tw=0
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if b[r,c]!=0 or vis[r,c]:continue
                stk=[(r,c)];reg=[];brd=set();vis[r,c]=1
                while stk:
                    rr,cc=stk.pop();reg.append((rr,cc))
                    for nr,nc in s._nbrs(rr,cc):
                        v=int(b[nr,nc])
                        if v==0 and not vis[nr,nc]:vis[nr,nc]=1;stk.append((nr,nc))
                        elif v!=0:brd.add(v)
                if brd=={1}:tb+=len(reg)
                elif brd=={-1}:tw+=len(reg)
        return(sb+tb)-(sw+tw+KOMI)
    def winner(s):return 1 if s.final_score_black()>0 else-1
def encode_state(st):
    b=st.board_array();own=(b==st.to_play).astype(np.float32);opp=(b==-st.to_play).astype(np.float32)
    return np.stack([own,opp,np.full_like(own,1.0 if st.to_play==1 else 0.0),np.zeros_like(own),np.full_like(own,min(st.move_count/MAX_GAME_LEN,1.0))],axis=0)

# === REGIME NETS ===
LOC_NAMES=["center","north","south","east","west","NE","NW","SE","SW"]
TACTIC_NAMES=["fight","pressure","open","solid","capture","defend","expand","reduce"]
PHASE_NAMES=["early","mid","late"]

class Regime:
    """A concept-recognizer: detects a type of board situation and contributes moves."""
    def __init__(self,rid,cr=3,cc=3,phase_lo=0.0,phase_hi=1.0):
        self.id=rid;self.name="unnamed_"+str(rid)
        self.center_r=cr;self.center_c=cc;self.radius=2.0
        self.phase_lo=phase_lo;self.phase_hi=phase_hi
        self.move_stats={};self.activations=0;self.total_value=0.0
        self.spatial_heat=np.zeros((BOARD_SIZE,BOARD_SIZE),dtype=np.float32)
        self.avg_density=0.0;self.avg_captures=0.0;self.birth_round=0
    def activation(self,board,to_play,move_count):
        """How strongly this regime recognizes the current position. Returns 0-1."""
        phase=move_count/MAX_GAME_LEN
        if phase<self.phase_lo-0.1 or phase>self.phase_hi+0.1:return 0.0
        phase_fit=1.0-min(1.0,max(0,abs(phase-(self.phase_lo+self.phase_hi)/2)-((self.phase_hi-self.phase_lo)/2))*3)
        # Spatial: how much action is near our center
        stones=(board!=0).astype(np.float32);density=stones.sum()/36
        dist=np.zeros((BOARD_SIZE,BOARD_SIZE),dtype=np.float32)
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):dist[r,c]=max(0,1.0-math.sqrt((r-self.center_r)**2+(c-self.center_c)**2)/max(0.5,self.radius))
        spatial_fit=float((stones*dist).sum())/max(1.0,float(stones.sum()))
        return min(1.0,phase_fit*0.5+spatial_fit*0.5+0.05)
    def policy_value(self):
        """Return (policy[37], value) from accumulated statistics."""
        pol=np.zeros(ALL_MOVES,dtype=np.float32)
        for mv,st in self.move_stats.items():
            if st["v"]>0:pol[mv]=st["w"]/st["v"]*math.log1p(st["v"])
        s=pol.sum()
        if s>0:pol/=s
        val=self.total_value/max(1,self.activations)
        return pol,np.clip(val,-1,1)
    def record(self,board,mv,outcome,to_play,move_count):
        """Record one position into this regime."""
        self.activations+=1;self.total_value+=outcome*to_play
        self.spatial_heat+=(board!=0).astype(np.float32)
        self.avg_density+=(((board!=0).sum()/36)-self.avg_density)/self.activations
        if mv not in self.move_stats:self.move_stats[mv]={"w":0,"v":0}
        self.move_stats[mv]["v"]+=1
        if outcome*to_play>0:self.move_stats[mv]["w"]+=1
    def auto_name(self):
        """Generate descriptive name from activation statistics."""
        cr,cc=self.center_r,self.center_c
        # Location
        if 2<=cr<=3 and 2<=cc<=3:loc="center"
        elif cr<2:loc="south" if cc<3 else"SE" if cc>=4 else"south"
        elif cr>=4:loc="north" if cc<3 else"NE" if cc>=4 else"north"
        else:loc="west" if cc<2 else"east" if cc>=4 else"mid"
        # Tactic from move patterns
        nmoves=len(self.move_stats);pass_rate=self.move_stats.get(PASS_MOVE,{}).get("v",0)/max(1,self.activations)
        if pass_rate>0.3:tactic="quiet"
        elif self.avg_density>0.5:tactic="fight"
        elif self.avg_density<0.15:tactic="open"
        elif nmoves>10:tactic="complex"
        else:tactic="solid"
        # Phase
        mid=(self.phase_lo+self.phase_hi)/2
        phase="early" if mid<0.3 else"late" if mid>0.7 else"mid"
        # Strength indicator
        wr=self.total_value/max(1,self.activations)
        star="" if abs(wr)<0.3 else"+" if wr>0 else"-"
        self.name=f"{loc}_{tactic}_{phase}{star}"

class RegimePool:
    """Population of regime nets — born, grow, merge, die during dreaming."""
    def __init__(self,max_regimes=32):
        self.regimes=[];self.max_regimes=max_regimes;self.lock=threading.Lock()
        self.experience_buffer=deque(maxlen=10000)  # short-term memory
        self.dream_cycles=0;self.dream_log=[]
    def query(self,board,to_play,move_count):
        """Run all regimes, return aggregated policy/value/confidence + active names."""
        pol=np.zeros(ALL_MOVES,dtype=np.float32);vsum=0.0;wsum=0.0;active=[]
        with self.lock:
            for reg in self.regimes:
                a=reg.activation(board,to_play,move_count)
                if a>0.2:
                    rp,rv=reg.policy_value();pol+=rp*a;vsum+=rv*a;wsum+=a
                    active.append({"name":reg.name,"activation":round(a,3),"id":reg.id})
        s=pol.sum()
        if s>0:pol/=s
        val=vsum/wsum if wsum>0 else 0.0;conf=min(1.0,wsum/max(1,len(self.regimes))*2)
        return pol,val,conf,active
    def record(self,board,mv,outcome,to_play,move_count):
        """Record experience into matching regimes + buffer."""
        self.experience_buffer.append({"board":board.copy(),"mv":mv,"outcome":outcome,"tp":to_play,"mc":move_count})
        with self.lock:
            for reg in self.regimes:
                if reg.activation(board,to_play,move_count)>0.3:
                    reg.record(board,mv,outcome,to_play,move_count)
    def dream(self,min_activations=5,round_num=0):
        """Dreaming: birth, merge, prune, rename regimes from experience buffer."""
        with self.lock:
            before=len(self.regimes);births=0;merges=0;prunes=0
            # Phase 1: PRUNE dead regimes
            alive=[r for r in self.regimes if r.activations>=min_activations]
            prunes=len(self.regimes)-len(alive)
            # Phase 2: MERGE overlapping regimes
            merged=[];skip=set()
            for i,ra in enumerate(alive):
                if i in skip:continue
                for j in range(i+1,len(alive)):
                    if j in skip:continue
                    rb=alive[j]
                    dist=math.sqrt((ra.center_r-rb.center_r)**2+(ra.center_c-rb.center_c)**2)
                    phase_overlap=max(0,min(ra.phase_hi,rb.phase_hi)-max(ra.phase_lo,rb.phase_lo))
                    if dist<2.0 and phase_overlap>0.2:
                        # Merge b into a
                        ra.activations+=rb.activations;ra.total_value+=rb.total_value
                        ra.spatial_heat+=rb.spatial_heat;ra.avg_density=(ra.avg_density+rb.avg_density)/2
                        for mv,st in rb.move_stats.items():
                            if mv not in ra.move_stats:ra.move_stats[mv]={"w":0,"v":0}
                            ra.move_stats[mv]["w"]+=st["w"];ra.move_stats[mv]["v"]+=st["v"]
                        skip.add(j);merges+=1
                merged.append(ra)
            # Phase 3: BIRTH new regimes from uncovered experience
            exps=list(self.experience_buffer)
            if len(exps)>10 and len(merged)<self.max_regimes:
                # Cluster by quadrant x phase
                for qr in[1,4]:
                    for qc in[1,4]:
                        for plo,phi in[(0,.33),(.33,.66),(.66,1.0)]:
                            matching=[e for e in exps if abs(e["board"][qr,qc])>=0 and plo<=e["mc"]/MAX_GAME_LEN<=phi and math.sqrt((qr-3)**2+(qc-3)**2)<4]
                            if len(matching)<3:continue
                            covered=any(r.activation(matching[0]["board"],matching[0]["tp"],matching[0]["mc"])>0.4 for r in merged)
                            if not covered and len(merged)<self.max_regimes:
                                nr=Regime(len(merged)+births+100,qr,qc,plo,phi);nr.birth_round=round_num
                                for e in matching[:50]:nr.record(e["board"],e["mv"],e["outcome"],e["tp"],e["mc"])
                                merged.append(nr);births+=1
            # Phase 4: RENAME all
            for r in merged:r.auto_name()
            # Phase 5: TRIM move stats
            for r in merged:
                if len(r.move_stats)>12:
                    top=sorted(r.move_stats.items(),key=lambda x:x[1]["v"],reverse=True)[:12]
                    r.move_stats=dict(top)
            self.regimes=merged;self.dream_cycles+=1
            info={"before":before,"after":len(merged),"births":births,"merges":merges,"prunes":prunes,
                  "cycle":self.dream_cycles,"names":[r.name for r in merged[:10]]}
            self.dream_log.append(info)
            self.experience_buffer.clear()
            return info
    def stats(self):
        with self.lock:
            return{"num_regimes":len(self.regimes),"dream_cycles":self.dream_cycles,
                   "buffer_size":len(self.experience_buffer),"dream_log":list(self.dream_log[-5:]),
                   "regimes":[{"name":r.name,"activations":r.activations,"avg_val":round(r.total_value/max(1,r.activations),3),"moves":len(r.move_stats)} for r in sorted(self.regimes,key=lambda x:x.activations,reverse=True)[:12]]}

REGIMES=RegimePool()

# === NEURAL NET ===
if not MOCK_MODE:
    class ResBlock(nn.Module):
        def __init__(s,ch):super().__init__();s.c1=nn.Conv2d(ch,ch,3,padding=1);s.b1=nn.BatchNorm2d(ch);s.c2=nn.Conv2d(ch,ch,3,padding=1);s.b2=nn.BatchNorm2d(ch)
        def forward(s,x):return F.relu(x+s.b2(s.c2(F.relu(s.b1(s.c1(x))))))
    class ExpertTower(nn.Module):
        def __init__(s,ch,dd,blk=1):
            super().__init__();s.blocks=nn.Sequential(*[ResBlock(ch) for _ in range(blk)]);s.ph=nn.Sequential(nn.Conv2d(ch,2,1),nn.BatchNorm2d(2),nn.ReLU());s.pf=nn.Linear(2*36,ALL_MOVES)
            s.vh=nn.Sequential(nn.Conv2d(ch,1,1),nn.BatchNorm2d(1),nn.ReLU());s.vf1=nn.Linear(36,48);s.vf2=nn.Linear(48,1);s.df=nn.Linear(ch,dd)
        def forward(s,base):h=s.blocks(base);return s.pf(s.ph(h).flatten(1)),torch.tanh(s.vf2(F.relu(s.vf1(s.vh(h).flatten(1))))).squeeze(-1),torch.tanh(s.df(F.adaptive_avg_pool2d(h,1).flatten(1)))
    class KGExpertNode(nn.Module):
        def __init__(s,dd=32):super().__init__();s.pn=nn.Sequential(nn.Linear(ALL_MOVES+2,64),nn.ReLU(),nn.Linear(64,ALL_MOVES));s.vn=nn.Sequential(nn.Linear(ALL_MOVES+2,32),nn.ReLU(),nn.Linear(32,1));s.dn=nn.Linear(ALL_MOVES+2,dd)
        def forward(s,f):return s.pn(f),torch.tanh(s.vn(f)).squeeze(-1),torch.tanh(s.dn(f))
    class GraphTeamNet(nn.Module):
        def __init__(s,ch=24,sb=1,eb=1,dd=32):
            super().__init__();ne=len(EXPERT_NAMES);s.expert_names=list(EXPERT_NAMES)
            s.stem=nn.Sequential(nn.Conv2d(5,ch,3,padding=1),nn.BatchNorm2d(ch),nn.ReLU());s.shared=nn.Sequential(*[ResBlock(ch) for _ in range(sb)])
            s.experts=nn.ModuleList([ExpertTower(ch,dd,eb) for _ in range(ne-1)]);s.kg_expert=KGExpertNode(dd)
            s.expert_token=nn.Parameter(torch.randn(ne,dd)*0.05);s.router=nn.Sequential(nn.Linear(ch+3,64),nn.ReLU(),nn.Linear(64,ne))
            s.edge_logits=nn.Parameter(torch.zeros(ne,ne));s.conf_head=nn.Linear(dd,1);s.router_temp=nn.Parameter(torch.tensor(1.0))
        def graph_matrix(s):return torch.softmax(s.edge_logits,dim=-1)
        def forward(s,x,kf=None,return_aux=False,expert_dropout=False):
            ne=len(s.expert_names);base=s.shared(s.stem(x));pool=F.adaptive_avg_pool2d(base,1).flatten(1)
            rl=s.router(torch.cat([pool,x[:,4].mean((1,2)).unsqueeze(-1),x[:,3].mean((1,2)).unsqueeze(-1),x[:,2].mean((1,2)).unsqueeze(-1)],-1))
            ps,vs,ds=[],[],[]
            for i,exp in enumerate(s.experts):lo,va,de=exp(base);ps.append(lo);vs.append(va);ds.append(de+s.expert_token[i].unsqueeze(0))
            if kf is None:kf=torch.zeros(x.shape[0],ALL_MOVES+2,device=x.device)
            kl,kv,kd=s.kg_expert(kf);ps.append(kl);vs.append(kv);ds.append(kd+s.expert_token[ne-1].unsqueeze(0))
            PS=torch.stack(ps,1);VS=torch.stack(vs,1);DS=torch.stack(ds,1)
            edges=s.graph_matrix();md=torch.einsum("ij,bjd->bid",edges,DS);conf=s.conf_head(torch.tanh(DS+md)).squeeze(-1)
            # Temperature floor at 1.0 to prevent sharpening collapse
            temp=torch.clamp(s.router_temp.abs(),1.0,3.0)
            w=torch.softmax((rl+conf)/temp,dim=-1)
            # Expert dropout: during training, randomly mask the dominant expert 30% of the time
            if expert_dropout and s.training:
                mask=torch.ones_like(w)
                top_idx=w.detach().argmax(dim=-1)  # [batch]
                drop=torch.rand(w.shape[0],device=w.device)<0.3  # 30% of batch
                for bi in range(w.shape[0]):
                    if drop[bi]:mask[bi,top_idx[bi]]=0.0
                w=w*mask;wsum=w.sum(-1,keepdim=True).clamp(min=1e-8);w=w/wsum
            fp=(PS*w.unsqueeze(-1)).sum(1);fv=(VS*w).sum(1)
            if not return_aux:return fp,fv
            return fp,fv,{"weights":w,"conf":conf}
        def snapshot_graph(s):
            with torch.no_grad():return{"experts":list(s.expert_names),"edges":s.graph_matrix().cpu().numpy().tolist(),"temp":float(torch.clamp(s.router_temp.abs(),1.0,3.0).cpu())}
    def new_team(d):n=GraphTeamNet();n.to(d);n.eval();return n
    def net_to_b64(n):b=io.BytesIO();torch.save(n.state_dict(),b);return base64.b64encode(gzip.compress(b.getvalue())).decode("ascii")
    def net_from_b64(p,d):n=new_team(d);n.load_state_dict(torch.load(io.BytesIO(gzip.decompress(base64.b64decode(p.encode("ascii")))),map_location=d,weights_only=False));n.eval();return n
    def _make_kf(state,device):
        b=state.board_array();scores,val,conf,_=REGIMES.query(b,state.to_play,state.move_count)
        return torch.from_numpy(np.concatenate([scores,[val,conf]])).float().unsqueeze(0).to(device)
    def infer_aux(net,device,state):
        x=torch.from_numpy(encode_state(state)).unsqueeze(0).to(device);kf=_make_kf(state,device)
        with torch.no_grad():lo,va,aux=net(x,kf=kf,return_aux=True)
        return lo.squeeze(0).cpu().numpy(),float(va.squeeze(0).cpu()),{"weights":aux["weights"].squeeze(0).cpu().numpy().tolist(),"conf":aux["conf"].squeeze(0).cpu().numpy().tolist()}

# === MCTS + SELFPLAY ===
@dataclass
class TreeNode:
    prior:float;to_play:int;vc:int=0;vs:float=0.0;children:Dict[int,"TreeNode"]=field(default_factory=dict);exp:bool=False
    def value(s):return 0.0 if s.vc==0 else s.vs/s.vc
if not MOCK_MODE:
    class MCTS:
        def __init__(s,net,dev,sims=32,c=1.5):s.net=net;s.dev=dev;s.sims=sims;s.c=c
        def _eval(s,st):
            x=torch.from_numpy(encode_state(st)).unsqueeze(0).to(s.dev);kf=_make_kf(st,s.dev)
            with torch.no_grad():lo,va=s.net(x,kf=kf);lo=lo.squeeze(0).cpu().numpy();val=float(va.squeeze(0).cpu())
            legal=st.legal_moves();mask=np.zeros(ALL_MOVES,dtype=np.float32);mask[legal]=1.0;lo[mask==0]=-1e9;pr=np.exp(lo-lo.max());pr*=mask;sm=pr.sum()
            return(mask/max(mask.sum(),1.0) if sm<=0 else pr/sm),val
        def _expand(s,nd,st,noise=False):
            pr,val=s._eval(st);legal=st.legal_moves()
            if noise and legal:n=np.random.dirichlet([0.3]*len(legal));[pr.__setitem__(legal[i],0.75*pr[legal[i]]+0.25*n[i]) for i in range(len(legal))]
            for mv in legal:
                cs=st.try_play(mv)
                if cs:nd.children[mv]=TreeNode(float(pr[mv]),cs.to_play)
            nd.exp=True;return val
        def _sel(s,nd):
            tv=math.sqrt(max(1,nd.vc));bs=-1e9;bm=PASS_MOVE;bc=None
            for mv,ch in nd.children.items():
                sc=-ch.value()+s.c*ch.prior*tv/(1+ch.vc)
                if sc>bs:bs=sc;bm=mv;bc=ch
            return bm,bc
        def run(s,st):
            root=TreeNode(1.0,st.to_play)
            if st.game_over():v=np.zeros(ALL_MOVES,dtype=np.float32);v[PASS_MOVE]=1.0;return v
            s._expand(root,st,noise=True)
            for _ in range(s.sims):
                nd=root;state=st;path=[nd]
                while nd.exp and nd.children:
                    mv,ch=s._sel(nd);ns=state.try_play(mv)
                    if not ns:break
                    nd=ch;state=ns;path.append(nd)
                    if state.game_over():break
                val=(1.0 if state.winner()==state.to_play else-1.0) if state.game_over() else s._expand(nd,state)
                for bn in reversed(path):bn.vc+=1;bn.vs+=val;val=-val
            vis=np.zeros(ALL_MOVES,dtype=np.float32)
            for mv,ch in root.children.items():vis[mv]=ch.vc
            return vis
    def sample_move(vis,st,temp):
        legal=st.legal_moves();pr=np.zeros_like(vis);pr[legal]=vis[legal]
        if pr.sum()<=0:pr[legal]=1.0
        if temp<=1e-4:mv=int(np.argmax(pr));oh=np.zeros_like(pr);oh[mv]=1.0;return mv,oh
        pp=pr**(1.0/temp);s=pp.sum()
        if s<=0:pp[legal]=1.0;s=pp.sum()
        pp/=s;return int(np.random.choice(np.arange(ALL_MOVES),p=pp)),pp
    def eval_move(net,dev,st,sims):
        vis=MCTS(net,dev,sims,1.3).run(st);legal=st.legal_moves();m=np.zeros_like(vis);m[legal]=vis[legal]
        return PASS_MOVE if m.sum()<=0 else int(np.argmax(m))
@dataclass
class Sample:
    sp:np.ndarray;pol:np.ndarray;z:float
class ReplayBuffer:
    def __init__(s,cap=50000):s.cap=cap;s.data=[];s.lock=threading.Lock()
    def add(s,batch):
        with s.lock:s.data.extend(batch);s.data=s.data[-s.cap:] if len(s.data)>s.cap else s.data
    def size(s):
        with s.lock:return len(s.data)
    def sample_batch(s,bs):
        with s.lock:
            if len(s.data)<bs:return None
            idx=np.random.choice(len(s.data),bs,replace=False);ch=[s.data[i] for i in idx]
        return np.stack([x.sp for x in ch]),np.stack([x.pol for x in ch]),np.array([x.z for x in ch],dtype=np.float32)
if not MOCK_MODE:
    def self_play_game(net,dev,sims,rn=0):
        st=GoState.new();samps=[];mvs=[];boards=[];tps=[];gt=[];mcts=MCTS(net,dev,sims)
        while not st.game_over():
            _,_,aux=infer_aux(net,dev,st);gt.append(aux["weights"])
            vis=mcts.run(st);temp=1.0 if st.move_count<10 else 1e-6;mv,pol=sample_move(vis,st,temp)
            samps.append((encode_state(st),pol.astype(np.float32),st.to_play));boards.append(st.board_array());mvs.append(mv);tps.append(st.to_play)
            ns=st.try_play(mv)
            if not ns:ns=st.try_play(PASS_MOVE)
            if not ns:break
            st=ns
        w=st.winner()
        for b,mv,tp in zip(boards,mvs,tps):REGIMES.record(b,mv,w,tp,0)
        return{"winner":w,"num_moves":len(mvs),"moves":[move_to_str(m) for m in mvs],"samples":[Sample(s,p,1.0 if pl==w else-1.0) for s,p,pl in samps],"avg_gate":np.mean(gt,axis=0).tolist() if gt else[0.0]*5}
    def do_selfplay_job(mb,dev,sims,games,rn=0):
        net=net_from_b64(mb,dev);results=[];packed=[]
        for _ in range(games):
            res=self_play_game(net,dev,sims,rn);results.append({k:v for k,v in res.items() if k!="samples"})
            for s in res["samples"]:packed.append([s.sp.tolist(),s.pol.tolist(),float(s.z)])
        return{"results":results,"samples":compress_obj(packed),"regime_stats":REGIMES.stats()}
    def train_team(net,replay,dev,steps=100,bs=64,lr=1e-3,lb_coeff=0.1):
        if replay.size()<bs:return{"steps":0,"total_loss":None}
        ne=len(EXPERT_NAMES)
        net.train();opt=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=1e-4);tl=[];bl=[]
        for _ in range(steps):
            b=replay.sample_batch(bs)
            if not b:break
            x,pt,z=[torch.from_numpy(a).to(dev) for a in b]
            # Forward with aux + expert dropout
            lo,va,aux=net(x,return_aux=True,expert_dropout=True)
            ploss=-(pt*F.log_softmax(lo,-1)).sum(-1).mean()
            vloss=F.mse_loss(va,z)
            # Load-balancing loss: penalize concentration
            # Switch Transformer style: ne * sum(fraction_i * router_prob_i)
            # Drives toward uniform 1/ne usage across batch
            w=aux["weights"]  # [batch, ne]
            frac=w.mean(0)  # avg weight per expert across batch
            # Ideal: each expert gets 1/ne fraction
            # Penalty: squared deviation from uniform
            lb_loss=ne*(frac*frac).sum()  # minimized when frac=[1/ne,...,1/ne]
            loss=ploss+vloss+lb_coeff*lb_loss
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.0);opt.step()
            tl.append(float((ploss+vloss).cpu()));bl.append(float(lb_loss.cpu()))
        net.eval()
        return{"steps":len(tl),"total_loss":float(np.mean(tl)) if tl else None,
               "balance_loss":float(np.mean(bl)) if bl else None,"graph":net.snapshot_graph()}
    def evaluate_pair(nets,dev,sims,games):
        def pm(nb,nw):
            st=GoState.new()
            while not st.game_over():
                mv=eval_move(nb if st.to_play==1 else nw,dev,st,sims);ns=st.try_play(mv)
                if not ns:ns=st.try_play(PASS_MOVE)
                if not ns:break
                st=ns
            return st.winner()
        wa=sum(1 for _ in range(games//2) if pm(nets["A"],nets["B"])==1)+sum(1 for _ in range(games-games//2) if pm(nets["B"],nets["A"])==-1)
        def vr(net,g=4):
            wins=0
            for _ in range(g):
                st=GoState.new()
                while not st.game_over():
                    mv=eval_move(net,dev,st,sims) if st.to_play==1 else random.choice(st.legal_moves())
                    ns=st.try_play(mv)
                    if not ns:ns=st.try_play(PASS_MOVE)
                    if not ns:break
                    st=ns
                if st.winner()==1:wins+=1
            return round(wins/max(1,g),3)
        return{"games":games,"wins_A":wa,"wins_B":games-wa,"A_vs_random":vr(nets["A"],max(4,games//2)),"B_vs_random":vr(nets["B"],max(4,games//2))}

class Leaderboard:
    def __init__(s):s.ratings={}
    def ensure(s,n,r=1000.0):s.ratings.setdefault(n,r)
    def update(s,a,b,sa,k=24.0):s.ensure(a);s.ensure(b);ea=1/(1+10**((s.ratings[b]-s.ratings[a])/400));s.ratings[a]+=k*(sa-ea);s.ratings[b]+=k*((1-sa)-(1-ea))
    def top(s):return sorted(s.ratings.items(),key=lambda x:x[1],reverse=True)

# === PARALLEL DISPATCH ===
def push_parallel(mb,sims,total,urls,token,local,dev,timeout=600,rn=0):
    workers=(["__local__"] if local else[])+list(urls)
    if not workers:workers=["__local__"]
    gp=max(1,total//len(workers));rem=total-gp*len(workers)
    assign=[(workers[i],gp+(1 if i<rem else 0)) for i in range(len(workers))];assign=[(w,g) for w,g in assign if g>0];results=[]
    def go(w,ng):
        t0=time.time()
        if w=="__local__":
            if MOCK_MODE:
                time.sleep(0.3*ng)
                for gi in range(ng):
                    out=random.choice([1,-1])
                    for mi in range(random.randint(5,20)):
                        b=np.zeros((6,6),dtype=np.int8)
                        for _ in range(random.randint(0,min(mi+2,12))):b[random.randint(0,5),random.randint(0,5)]=random.choice([1,-1])
                        REGIMES.record(b,random.randint(0,35),out,random.choice([1,-1]),mi)
                return{"ok":True,"results":[{"winner":random.choice([1,-1])}]*ng,"samples":compress_obj([]),"worker_name":"local","seconds":round(time.time()-t0,2)}
            res=do_selfplay_job(mb,dev,sims,ng,rn);res["worker_name"]="local";res["seconds"]=round(time.time()-t0,2);return res
        try:return http_json(w.rstrip("/")+"/selfplay",method="POST",payload={"model":mb,"sims":sims,"games":ng,"round_num":rn},token=token,timeout=timeout)
        except:
            if MOCK_MODE:return{"ok":True,"results":[{"winner":1}]*ng,"samples":compress_obj([]),"worker_name":"fb("+w+")","seconds":round(time.time()-t0,2)}
            res=do_selfplay_job(mb,dev,sims,ng,rn);res["worker_name"]="fb("+w+")";res["seconds"]=round(time.time()-t0,2);return res
    with concurrent.futures.ThreadPoolExecutor(max(1,len(assign))) as ex:
        for f in concurrent.futures.as_completed({ex.submit(go,w,g):(w,g) for w,g in assign}):
            try:results.append(f.result())
            except Exception as e:results.append({"ok":False,"error":str(e),"worker_name":"err","results":[],"samples":compress_obj([]),"seconds":0})
    return results

# === APP STATE ===
class WorkerState:
    def __init__(s):s.jobs_done=0;s.total_games=0;s.busy=False;s.last_job_time=None;s.log=deque(maxlen=200);s.started_at=now_ts()
    def _log(s,msg):s.log.append({"ts":now_ts(),"msg":msg})
WS=WorkerState()
class AppState:
    def __init__(s):
        s.device=choose_device();s.nets={};s.replays={};s.lb=Leaderboard();s.worker_urls=[];s.token="";s.include_local=True
        s.cfg={"rounds":6,"selfplay_sims":20,"eval_sims":28,"games_per_job":2,"selfplay_jobs_per_round":2,"train_steps":60,"batch_size":64,"learning_rate":0.001,"eval_games":8,"dream_interval":5,"dream_min_act":5}
        s.log=deque(maxlen=500);s.errors=deque(maxlen=100);s.rh=[];s.cr=0;s.is_training=False;s._stop=False;s.pg_state=None;s.pg_moves=[]
    def _log(s,m):s.log.append({"ts":now_ts(),"msg":m})
    def _err(s,m,exc=None):
        entry={"ts":now_ts(),"msg":m}
        if exc:
            import traceback;entry["traceback"]=traceback.format_exc();entry["type"]=type(exc).__name__
        s.errors.append(entry);s.log.append({"ts":now_ts(),"msg":"[ERROR] "+m})
    def init_nets(s):
        for n in["A","B"]:s.nets[n]=new_team(s.device) if not MOCK_MODE else"mock_"+n;s.replays[n]=ReplayBuffer();s.lb.ensure(n+"_current")
        s._log("Nets init"+(" (demo)" if MOCK_MODE else ""))
    def bj(s,st):
        b=st.board_array();cells=[];lr=[]
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):cells.append({"r":r,"c":c,"v":int(b[r,c])})
        for mv in st.legal_moves():
            if mv==PASS_MOVE:lr.append({"r":-1,"c":-1,"move":mv})
            else:rr,cc=divmod(mv,BOARD_SIZE);lr.append({"r":rr,"c":cc,"move":mv})
        return{"size":BOARD_SIZE,"cells":cells,"to_play":st.to_play,"to_play_label":"Black" if st.to_play==1 else "White","passes":st.passes,"move_count":st.move_count,"game_over":st.game_over(),"legal_moves":lr,"score_black":st.final_score_black() if st.game_over() else None}
ST=AppState();APP_MODE="coordinator"

# === FLASK ===
from flask import Flask,Response,request,jsonify
app=Flask(__name__);app.secret_key=secrets.token_hex(16)
CSS=":root{--bg:#0b0d0e;--bg2:#121618;--bg3:#1a1e22;--brd:#262c32;--fg:#c5cad0;--fg2:#8a9099;--fg3:#585e66;--gold:#c9a227;--gold2:#e0be4a;--teal:#3aafa9;--rose:#d45d5d;--sky:#5b8fd4;--violet:#9b6fd4;--stone-b:#181818;--stone-w:#e6e2d8}*{margin:0;padding:0;box-sizing:border-box}body{background:var(--bg);color:var(--fg);font-family:'IBM Plex Mono',monospace;font-size:13px;line-height:1.55}::selection{background:var(--gold);color:var(--bg)}.top{position:sticky;top:0;z-index:100;background:var(--bg);border-bottom:1px solid var(--brd);display:flex;align-items:center;height:50px;padding:0 1.25rem;gap:1.5rem}.top h1{font-family:'Playfair Display',serif;font-size:1.15rem;font-weight:700;color:var(--gold);white-space:nowrap}.top h1 em{font-style:normal;color:var(--fg3);font-size:.7rem;margin-left:.5rem}.tabs{display:flex}.tabs button{background:0;border:0;color:var(--fg3);font:inherit;font-size:11.5px;font-weight:500;padding:.55rem 1rem;cursor:pointer;border-bottom:2px solid transparent}.tabs button:hover{color:var(--fg)}.tabs button.on{color:var(--gold);border-bottom-color:var(--gold)}.top-r{margin-left:auto;display:flex;align-items:center;gap:.7rem}.dot{width:7px;height:7px;border-radius:50%;background:var(--fg3);display:inline-block}.dot.ok{background:var(--teal)}.dot.run{background:var(--gold);animation:pls 1.1s infinite}@keyframes pls{0%,100%{opacity:1}50%{opacity:.3}}.pan{display:none;padding:1.25rem}.pan.on{display:block}.g2{display:grid;grid-template-columns:1fr 1fr;gap:1.1rem}@media(max-width:900px){.g2{grid-template-columns:1fr}}.c{background:var(--bg2);border:1px solid var(--brd);border-radius:5px;overflow:hidden;margin-bottom:1.1rem}.ch{padding:.55rem .9rem;border-bottom:1px solid var(--brd);font-size:10.5px;text-transform:uppercase;letter-spacing:.09em;color:var(--fg3)}.cb{padding:.85rem .9rem}.fld{display:flex;flex-direction:column;gap:2px;font-size:11px;color:var(--fg2);margin-bottom:.6rem}.fld input,.fld select{background:var(--bg3);border:1px solid var(--brd);color:var(--fg);font:inherit;font-size:12.5px;padding:5px 7px;border-radius:3px}.btn{display:inline-flex;align-items:center;background:var(--gold);color:var(--bg);font:inherit;font-size:11.5px;font-weight:600;padding:6px 13px;border:0;border-radius:3px;cursor:pointer}.btn:hover{background:var(--gold2)}.btn:disabled{opacity:.35}.btn.o{background:0;color:var(--gold);border:1px solid var(--gold)}.btn.o:hover{background:var(--gold);color:var(--bg)}.btn.d{background:var(--rose);color:#fff}.btn.s{font-size:10.5px;padding:4px 9px}.mt{width:100%;border-collapse:collapse}.mt td{padding:4px 7px;border-bottom:1px solid var(--brd);font-size:12px}.mt td:first-child{color:var(--fg3);width:44%}.mt td:last-child{font-weight:500}.eb{display:flex;align-items:center;gap:.45rem;margin:3px 0}.eb-l{width:66px;font-size:10.5px;color:var(--fg2);text-align:right}.eb-t{flex:1;height:15px;background:var(--bg3);border-radius:2px;overflow:hidden}.eb-f{height:100%;border-radius:2px;transition:width .35s}.eb-v{width:38px;font-size:10.5px;color:var(--fg3)}.ag{display:inline-grid;gap:2px;font-size:10px}.ac{width:42px;height:22px;display:flex;align-items:center;justify-content:center;border-radius:2px;font-weight:500;font-size:9px}.ah{color:var(--fg3);font-size:9px}.la{background:var(--bg);border:1px solid var(--brd);border-radius:3px;font-size:11px;height:220px;overflow-y:auto;padding:.4rem;color:var(--fg2)}.la .ts{color:var(--fg3);margin-right:.4rem}.lr{display:flex;align-items:center;gap:.5rem;padding:4px 0;border-bottom:1px solid var(--brd);font-size:12px}.lr-k{width:22px;color:var(--fg3);text-align:center}.lr-n{flex:1}.lr-e{font-weight:500;color:var(--gold);width:56px;text-align:right}.wr{display:flex;align-items:center;gap:.5rem;padding:5px 0;border-bottom:1px solid var(--brd);font-size:12px}.wr .u{flex:1}.ws{font-size:11px}.ws.ok{color:var(--teal)}.ws.er{color:var(--rose)}.em{color:var(--fg3);font-style:italic;font-size:12px;padding:.8rem 0}.badge{display:inline-block;font-size:9.5px;font-weight:600;text-transform:uppercase;padding:2px 7px;border-radius:2px}.badge.demo{background:rgba(212,93,93,.12);color:var(--rose)}.badge.live{background:rgba(58,175,169,.12);color:var(--teal)}.badge.work{background:rgba(91,143,212,.12);color:var(--sky)}.rg{display:inline-block;background:rgba(155,111,212,.1);border:1px solid rgba(155,111,212,.25);color:var(--violet);font-size:10.5px;padding:2px 8px;border-radius:3px;margin:2px}.rg .rv{color:var(--fg3);font-size:9.5px;margin-left:4px}.big-url{font-size:1.1rem;color:var(--gold);background:var(--bg3);border:1px dashed var(--gold);border-radius:4px;padding:.6rem 1rem;text-align:center;margin:.6rem 0;cursor:pointer;user-select:all}.step{display:flex;gap:.6rem;margin:.5rem 0}.step-n{background:var(--gold);color:var(--bg);width:22px;height:22px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;flex-shrink:0}.step-t{font-size:12px}"
FONT='<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=Playfair+Display:wght@400;700&display=swap" rel="stylesheet"/>'
def bld(title,bc,bt,tabs,body,js):return f"<!DOCTYPE html><html><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><title>{title}</title>{FONT}<style>{CSS}.bx-btn{{background:var(--bg3);border:1px solid var(--brd);color:var(--rose);font:inherit;font-size:10px;font-weight:600;padding:3px 8px;border-radius:3px;cursor:pointer;letter-spacing:.05em}}.bx-btn:hover{{background:var(--rose);color:#fff}}</style></head><body><div class=top><h1>EXOSFEAR <em>MicroGo KG v4</em></h1>{tabs}<div class=top-r><button class=bx-btn onclick=doBX()>BX</button><span class='badge {bc}'>{bt}</span><span class=dot id=sd></span><span id=sl style='font-size:11px;color:var(--fg3)'>idle</span></div></div>{body}<script>var _jsErrors=[];window.onerror=function(m,s,l,c,e){{_jsErrors.push({{msg:m,src:s,line:l,col:c,stack:e?e.stack:'',ts:new Date().toISOString()}});if(_jsErrors.length>50)_jsErrors.shift()}};async function doBX(){{try{{const r=await fetch('/api/bx',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{js_errors:_jsErrors}})}});const d=await r.json();d._url=location.href;d._ua=navigator.userAgent;d._screen=screen.width+'x'+screen.height;const b=new Blob([JSON.stringify(d,null,2)],{{type:'application/json'}});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='microgo_bx_'+Date.now()+'.json';a.click();document.querySelector('.bx-btn').style.borderColor='var(--teal)';setTimeout(()=>document.querySelector('.bx-btn').style.borderColor='',2000)}}catch(e){{alert('BX capture failed: '+e)}}}};{js}</script></body></html>"

def worker_page():
    return bld("EXOSFEAR Worker","work","WORKER","","""<div style="padding:1.25rem"><div class="g2"><div>
<div class="c"><div class="ch">Connection Info</div><div class="cb"><div class="big-url" id="wurl" onclick="navigator.clipboard.writeText(this.textContent)">loading...</div><p style="font-size:11px;color:var(--fg3)">Click to copy</p></div></div>
<div class="c"><div class="ch">Setup</div><div class="cb"><div class="step"><span class="step-n">1</span><span class="step-t">Worker running. Keep open.</span></div><div class="step"><span class="step-n">2</span><span class="step-t">Main: <code style="color:var(--gold)">python microgo.py coordinator</code></span></div><div class="step"><span class="step-n">3</span><span class="step-t">Add URL in Workers tab. Start training.</span></div></div></div>
</div><div><div class="c"><div class="ch">Status</div><div class="cb"><table class="mt"><tr><td>Status</td><td id="ws">idle</td></tr><tr><td>Device</td><td id="wdev">-</td></tr><tr><td>Jobs</td><td id="wj">0</td></tr><tr><td>Games</td><td id="wg">0</td></tr><tr><td>Regimes</td><td id="wrg">0</td></tr></table></div></div>
<div class="c"><div class="ch">Log</div><div class="cb" style="padding:0"><div class="la" id="wlog" style="height:300px"></div></div></div></div></div></div>""",
    r"""const $=id=>document.getElementById(id);const api=async(p,b)=>{const o=b!==undefined?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)}:{};return(await fetch(p,o)).json()};async function wp(){try{const d=await api('/api/worker/status');$('wurl').textContent=d.url;$('wdev').textContent=d.device;$('wj').textContent=d.jobs_done;$('wg').textContent=d.total_games;$('wrg').textContent=d.regimes;$('ws').textContent=d.busy?'WORKING':'idle';$('ws').style.color=d.busy?'var(--gold)':'var(--teal)';$('sd').className='dot '+(d.busy?'run':'ok');if(d.log&&d.log.length){const el=$('wlog');let h='';for(const e of d.log)h+=`<div><span class="ts">${e.ts}</span>${e.msg}</div>`;el.innerHTML=h;el.scrollTop=el.scrollHeight}}catch(e){}}wp();setInterval(wp,2000);""")

def coord_page():
    b="demo" if MOCK_MODE else "live"
    tabs='<div class="tabs"><button class="on" data-tab="dash">Dashboard</button><button data-tab="play">Playground</button><button data-tab="work">Workers</button><button data-tab="conf">Config</button></div>'
    body="""
<div class="pan on" id="t-dash"><div class="g2"><div>
<div class="c"><div class="ch">Training</div><div class="cb"><table class="mt"><tr><td>Round</td><td id="mr">0</td></tr><tr><td>Replay A / B</td><td id="mrab">0 / 0</td></tr><tr><td>Loss A / B</td><td id="mlab">-</td></tr><tr><td>vs Random</td><td id="mvr">-</td></tr><tr><td>H2H</td><td id="mhh">-</td></tr></table><div style="margin-top:.6rem;display:flex;gap:.4rem"><button class="btn" id="bs1" onclick="startT()">Start</button><button class="btn o" id="bs2" onclick="stopT()" disabled>Stop</button></div></div></div>
<div class="c"><div class="ch">Expert Routing (5 nodes)</div><div class="cb" id="ebs"><p class="em">Run training first</p></div></div>
<div class="c"><div class="ch">Regime Nets (memory node)</div><div class="cb" id="rgd"><p class="em">Regimes appear after dreaming</p></div></div>
<div class="c"><div class="ch">Graph Adjacency 5x5</div><div class="cb" id="adj"><p class="em">After first round</p></div></div>
</div><div>
<div class="c"><div class="ch">Leaderboard</div><div class="cb" id="lbd"><p class="em">No ratings</p></div></div>
<div class="c"><div class="ch">Loss History</div><div class="cb"><canvas id="lc" height="160" style="width:100%"></canvas></div></div>
<div class="c"><div class="ch">Log</div><div class="cb" style="padding:0"><div class="la" id="elog"></div></div></div>
</div></div></div>
<div class="pan" id="t-play"><div class="g2"><div><div class="c"><div class="ch">Board 6x6</div><div class="cb" style="display:flex;justify-content:center;padding:.8rem"><svg id="bs" viewBox="0 0 320 320" style="max-width:380px;width:100%"></svg></div><div class="cb" style="border-top:1px solid var(--brd)"><table class="mt"><tr><td>To Play</td><td id="pt">-</td></tr><tr><td>Move</td><td id="pm">0</td></tr><tr><td>Result</td><td id="pr">-</td></tr></table><div style="margin-top:.6rem;display:flex;gap:.4rem"><button class="btn s" onclick="pNew()">New</button><button class="btn s o" onclick="pPass()">Pass</button><button class="btn s o" onclick="pAI()">AI</button><button class="btn s o" onclick="pAuto()">Auto</button></div></div></div></div><div><div class="c"><div class="ch">Experts</div><div class="cb" id="pex"><p class="em">Start game</p></div></div><div class="c"><div class="ch">Active Regimes</div><div class="cb" id="par"><p class="em">-</p></div></div><div class="c"><div class="ch">Moves</div><div class="cb" id="pmv"><p class="em">No moves</p></div></div></div></div></div>
<div class="pan" id="t-work"><div class="g2"><div><div class="c"><div class="ch">Workers</div><div class="cb" id="wl"><p class="em">No workers</p></div><div class="cb" style="border-top:1px solid var(--brd)"><label class="fld">Add<div style="display:flex;gap:.35rem"><input type="text" id="nwu" placeholder="http://192.168.1.50:8765" style="flex:1"/><button class="btn s" onclick="addW()">Add</button></div></label><div style="display:flex;gap:.4rem"><button class="btn s o" onclick="pingW()">Ping</button><button class="btn s o" onclick="scanW()">Scan LAN</button></div></div></div></div><div><div class="c"><div class="ch">Cluster</div><div class="cb"><table class="mt"><tr><td>Device</td><td id="wd">-</td></tr><tr><td>IPs</td><td id="wi">-</td></tr><tr><td>Token</td><td id="wt">-</td></tr></table></div></div></div></div></div>
<div class="pan" id="t-conf"><div class="g2"><div><div class="c"><div class="ch">Training</div><div class="cb"><label class="fld">Rounds<input type="number" id="cr" value="6" min="1"/></label><label class="fld">Sims<input type="number" id="cs" value="20"/></label><label class="fld">Games/job<input type="number" id="cg" value="2"/></label><label class="fld">Jobs/round<input type="number" id="cj" value="2"/></label><label class="fld">Train steps<input type="number" id="ct" value="60"/></label><label class="fld">Batch<input type="number" id="ccb" value="64"/></label><label class="fld">LR<input type="text" id="cl" value="0.001"/></label><button class="btn" onclick="saveCfg()">Save</button></div></div></div><div><div class="c"><div class="ch">Dreaming</div><div class="cb"><label class="fld">Dream every N rounds<input type="number" id="cdi" value="5"/></label><label class="fld">Min activations<input type="number" id="cdm" value="5"/></label><button class="btn" onclick="saveDr()">Save</button><button class="btn o" style="margin-left:.4rem" onclick="api('/api/dream',{})">Dream Now</button></div></div><div class="c"><div class="ch">Actions</div><div class="cb"><button class="btn o" onclick="resetN()">Reset</button></div></div></div></div></div>"""
    js=r"""const $=id=>document.getElementById(id);const api=async(p,b)=>{const o=b!==undefined?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)}:{};return(await fetch(p,o)).json()};
document.querySelector('.tabs').addEventListener('click',e=>{if(e.target.tagName!=='BUTTON')return;const id=e.target.dataset.tab;document.querySelectorAll('.pan').forEach(p=>p.classList.remove('on'));document.querySelectorAll('.tabs button').forEach(b=>b.classList.remove('on'));document.getElementById('t-'+id).classList.add('on');e.target.classList.add('on');if(id==='play')pNew()});
function eBars(ew,tgt){if(!ew||!ew.length)return;const cols=['var(--gold)','var(--sky)','var(--teal)','var(--rose)','var(--violet)'];const ns=['opening','tactics','territory','endgame','memory'];let h='';for(let i=0;i<ew.length;i++){const p=(ew[i]*100).toFixed(1);h+=`<div class="eb"><span class="eb-l">${ns[i]||'?'}</span><div class="eb-t"><div class="eb-f" style="width:${p}%;background:${cols[i]}"></div></div><span class="eb-v">${p}%</span></div>`}$(tgt).innerHTML=h}
async function poll(){try{const d=await api('/api/status');$('mr').textContent=d.cr+' / '+d.cfg.rounds;$('mrab').textContent=d.ra+' / '+d.rb;$('mlab').textContent=(d.la||'-')+' / '+(d.lb||'-');$('mvr').textContent='A:'+(d.avr||'-')+' B:'+(d.bvr||'-');$('mhh').textContent=d.h2h||'-';
if(d.training){$('sd').className='dot run';$('sl').textContent='training';$('bs1').disabled=true;$('bs2').disabled=false}else{$('sd').className='dot ok';$('sl').textContent='idle';$('bs1').disabled=false;$('bs2').disabled=true}
if(d.lb_data&&d.lb_data.length){let h='';for(let i=0;i<d.lb_data.length;i++){const e=d.lb_data[i];h+=`<div class="lr"><span class="lr-k">${i+1}</span><span class="lr-n">${e[0]}</span><span class="lr-e">${Math.round(e[1])}</span></div>`}$('lbd').innerHTML=h}
eBars(d.ew,'ebs');
if(d.regimes&&d.regimes.regimes&&d.regimes.regimes.length){let h=`<div style="font-size:11px;color:var(--fg3);margin-bottom:6px">${d.regimes.num_regimes} regimes, ${d.regimes.dream_cycles} dreams, buf:${d.regimes.buffer_size}</div>`;for(const r of d.regimes.regimes){h+=`<span class="rg">${r.name}<span class="rv">${r.activations}x val:${r.avg_val}</span></span>`}$('rgd').innerHTML=h}
if(d.graph&&d.graph.edges){const ns=d.graph.experts,n=ns.length;let h=`<div class="ag" style="grid-template-columns:50px repeat(${n},42px)"><div></div>`;for(const nm of ns)h+=`<div class="ah">${nm.slice(0,4)}</div>`;for(let ri=0;ri<n;ri++){h+=`<div class="ah" style="text-align:right;padding-right:3px">${ns[ri].slice(0,4)}</div>`;for(const v of d.graph.edges[ri])h+=`<div class="ac" style="background:rgba(201,162,39,${Math.min(1,v*2.5).toFixed(2)})">${v.toFixed(2)}</div>`}$('adj').innerHTML=h+'</div>'}
if(d.log&&d.log.length){const el=$('elog');let h='';for(const e of d.log)h+=`<div><span class="ts">${e.ts}</span>${e.msg}</div>`;el.innerHTML=h;el.scrollTop=el.scrollHeight}
if(d.rh&&d.rh.length){const cv=$('lc'),ctx=cv.getContext('2d');const dpr=devicePixelRatio||1;cv.width=cv.offsetWidth*dpr;cv.height=160*dpr;ctx.scale(dpr,dpr);const w=cv.offsetWidth,h=160;ctx.clearRect(0,0,w,h);const la=d.rh.map(r=>r.la).filter(v=>v!=null),lb=d.rh.map(r=>r.lb).filter(v=>v!=null);if(la.length){const all=[...la,...lb],mn=Math.min(...all),mx=Math.max(...all),rng=mx-mn||1,pad={t:8,r:8,b:20,l:34},pw=w-pad.l-pad.r,ph=h-pad.t-pad.b;ctx.strokeStyle='#262c32';ctx.lineWidth=1;for(let i=0;i<=4;i++){const y=pad.t+ph*i/4;ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(w-pad.r,y);ctx.stroke()}function line(data,col){ctx.strokeStyle=col;ctx.lineWidth=2;ctx.beginPath();data.forEach((v,i)=>{const x=pad.l+(i/Math.max(1,data.length-1))*pw,y=pad.t+(1-(v-mn)/rng)*ph;i===0?ctx.moveTo(x,y):ctx.lineTo(x,y)});ctx.stroke()}line(la,'#c9a227');line(lb,'#5b8fd4')}}}catch(e){}}
async function startT(){await api('/api/train/start',{})}async function stopT(){await api('/api/train/stop',{})}async function resetN(){if(confirm('Reset?'))await api('/api/reset',{})}
async function saveCfg(){await api('/api/config',{rounds:+$('cr').value,selfplay_sims:+$('cs').value,games_per_job:+$('cg').value,selfplay_jobs_per_round:+$('cj').value,train_steps:+$('ct').value,batch_size:+$('ccb').value,learning_rate:+$('cl').value})}
async function saveDr(){await api('/api/config',{dream_interval:+$('cdi').value,dream_min_act:+$('cdm').value})}
async function addW(){const u=$('nwu').value.trim();if(u){await api('/api/workers/add',{url:u});$('nwu').value='';rfW()}}
async function pingW(){await api('/api/workers/ping',{});rfW()}async function scanW(){$('wl').innerHTML='<p class="em">Scanning...</p>';await api('/api/workers/scan',{});rfW()}
async function rfW(){const d=await api('/api/workers');$('wd').textContent=d.device;$('wi').textContent=(d.ips||[]).join(', ')||'-';$('wt').textContent=d.token?d.token.slice(-4):'-';const el=$('wl');if(!d.workers||!d.workers.length){el.innerHTML='<p class="em">No workers</p>';return}let h='';for(const w of d.workers)h+=`<div class="wr"><span class="dot ${w.ok?'ok':''}"></span><span class="u">${w.url}</span><span class="ws ${w.ok?'ok':'er'}">${w.ok?'online':'off'}</span><button class="btn s d" data-rm="${w.url}">x</button></div>`;el.innerHTML=h}
document.addEventListener('click',e=>{if(e.target.dataset.rm){api('/api/workers/remove',{url:e.target.dataset.rm}).then(()=>rfW())}});
async function pNew(){const d=await api('/api/playground/new',{});rB(d)}async function pPass(){const d=await api('/api/playground/move',{move:'pass'});rB(d)}async function pAI(){const d=await api('/api/playground/ai',{});rB(d)}async function pAuto(){for(let i=0;i<60;i++){const d=await api('/api/playground/ai',{});rB(d);if(d.board.game_over)break;await new Promise(r=>setTimeout(r,180))}}
function rB(data){if(!data||!data.board)return;const b=data.board,svg=$('bs'),S=b.size,mg=28,cs=(320-2*mg)/(S-1);let h=`<rect x="4" y="4" width="312" height="312" rx="4" fill="#b0862a"/><rect x="7" y="7" width="306" height="306" rx="3" fill="#c08c32"/>`;for(let i=0;i<S;i++){const p=mg+i*cs;h+=`<line x1="${mg}" y1="${p}" x2="${320-mg}" y2="${p}" stroke="#896320" stroke-width=".8"/><line x1="${p}" y1="${mg}" x2="${p}" y2="${320-mg}" stroke="#896320" stroke-width=".8"/>`}[[1,1],[1,4],[4,1],[4,4]].forEach(([r,c])=>{h+=`<circle cx="${mg+c*cs}" cy="${mg+r*cs}" r="2.3" fill="#896320"/>`});for(const c of b.cells){if(c.v===0)continue;const cx=mg+c.c*cs,cy=mg+c.r*cs,rd=cs*.41;h+=c.v===1?`<circle cx="${cx}" cy="${cy}" r="${rd}" fill="#181818" stroke="#2a2a2a" stroke-width=".5"/>`:`<circle cx="${cx}" cy="${cy}" r="${rd}" fill="#e6e2d8" stroke="#bbb" stroke-width=".5"/>`}if(!b.game_over)for(const lm of b.legal_moves){if(lm.r<0)continue;const cx=mg+lm.c*cs,cy=mg+lm.r*cs;h+=`<circle cx="${cx}" cy="${cy}" r="${cs*.28}" fill="transparent" stroke="${b.to_play===1?'#444':'#ccc'}" stroke-width="1" stroke-dasharray="3,3" opacity="0" cursor="pointer" onmouseover="this.setAttribute('opacity','.45')" onmouseout="this.setAttribute('opacity','0')" data-mv="${lm.move}"/>`}for(let i=0;i<S;i++){const x=mg+i*cs;h+=`<text x="${x}" y="15" text-anchor="middle" fill="#896320" font-size="10">${String.fromCharCode(65+i)}</text><text x="11" y="${mg+i*cs+4}" text-anchor="middle" fill="#896320" font-size="10">${i+1}</text>`}svg.innerHTML=h;$('pt').textContent=b.to_play_label;$('pm').textContent=b.move_count;$('pr').textContent=b.game_over?(b.score_black>0?'B+'+b.score_black.toFixed(1):'W+'+(-b.score_black).toFixed(1)):'-';eBars(data.experts,'pex');
if(data.active_regimes&&data.active_regimes.length){let rh='';for(const r of data.active_regimes)rh+=`<span class="rg">${r.name}<span class="rv">${(r.activation*100).toFixed(0)}%</span></span>`;$('par').innerHTML=rh}else $('par').innerHTML='<p class="em">No regimes active</p>';
if(data.moves&&data.moves.length){let mh='<div style="display:flex;flex-wrap:wrap;gap:3px">';for(let i=0;i<data.moves.length;i++)mh+=`<span class="rg" style="background:var(--bg3);border-color:var(--brd);color:var(--fg)">${i+1}.${data.moves[i]}</span>`;$('pmv').innerHTML=mh+'</div>'}else $('pmv').innerHTML='<p class="em">No moves</p>'}
document.getElementById('bs').addEventListener('click',e=>{if(e.target.dataset.mv)api('/api/playground/move',{move:+e.target.dataset.mv}).then(d=>rB(d))});
poll();setInterval(poll,3000);rfW();"""
    return bld("EXOSFEAR MicroGo KG v4",b,b.upper(),tabs,body,js)

# === ROUTES ===
@app.route("/")
def index():return Response(worker_page() if APP_MODE=="worker" else coord_page(),mimetype="text/html")
@app.route("/health")
def health():return jsonify({"ok":True,"role":"microgo_worker","worker_name":socket.gethostname(),"device":ST.device,"version":VERSION})
@app.route("/selfplay",methods=["POST"])
def selfplay_ep():
    tk=app.config.get("AUTH_TOKEN","")
    if tk and request.headers.get("X-Token","")!=tk:return jsonify({"ok":False}),403
    data=request.json or{};WS.busy=True;WS._log("Job "+str(data.get("games",0))+"g");t0=time.time()
    try:
        if MOCK_MODE:
            ng=int(data.get("games",2));time.sleep(0.4*ng)
            for gi in range(ng):
                out=random.choice([1,-1])
                for mi in range(random.randint(5,15)):
                    b=np.zeros((6,6),dtype=np.int8)
                    for _ in range(random.randint(0,min(mi+2,10))):b[random.randint(0,5),random.randint(0,5)]=random.choice([1,-1])
                    REGIMES.record(b,random.randint(0,35),out,random.choice([1,-1]),mi)
            res={"ok":True,"results":[{"winner":random.choice([1,-1])}]*ng,"samples":compress_obj([]),"worker_name":socket.gethostname(),"seconds":round(time.time()-t0,2)}
        else:
            res=do_selfplay_job(data["model"],ST.device,int(data.get("sims",20)),int(data.get("games",2)),int(data.get("round_num",0)))
            res["ok"]=True;res["worker_name"]=socket.gethostname();res["seconds"]=round(time.time()-t0,2)
        WS.jobs_done+=1;WS.total_games+=len(res.get("results",[]));WS.last_job_time=now_ts();WS._log("Done "+str(len(res.get("results",[])))+"g "+str(res["seconds"])+"s");return jsonify(res)
    except Exception as e:WS._log("FAIL:"+str(e));ST._err("Selfplay: "+str(e),e);return jsonify({"ok":False,"error":str(e)}),500
    finally:WS.busy=False
@app.route("/api/worker/status")
def wk_st():
    ips=get_local_ips();p=app.config.get("WORKER_PORT",DEFAULT_WORKER_PORT)
    return jsonify({"url":("http://"+ips[0]+":"+str(p)) if ips else"","device":ST.device,"token":app.config.get("AUTH_TOKEN",""),"busy":WS.busy,"jobs_done":WS.jobs_done,"total_games":WS.total_games,"last_job_time":WS.last_job_time,"regimes":REGIMES.stats()["num_regimes"],"log":list(WS.log)})
@app.route("/api/status")
def api_status():
    last=ST.rh[-1] if ST.rh else{}
    return jsonify({"training":ST.is_training,"cr":ST.cr,"cfg":ST.cfg,"ra":ST.replays["A"].size() if"A"in ST.replays else 0,"rb":ST.replays["B"].size() if"B"in ST.replays else 0,"la":last.get("la"),"lb":last.get("lb"),"avr":last.get("avr"),"bvr":last.get("bvr"),"h2h":last.get("h2h"),"ew":last.get("ew"),"graph":last.get("graph"),"lb_data":ST.lb.top()[:10],"rh":ST.rh,"log":list(ST.log),"regimes":REGIMES.stats()})
@app.route("/api/config",methods=["POST"])
def api_cfg():
    d=request.json or{}
    for k in["rounds","selfplay_sims","games_per_job","selfplay_jobs_per_round","train_steps","batch_size","eval_sims","eval_games","dream_interval","dream_min_act"]:
        if k in d:ST.cfg[k]=int(d[k])
    if"learning_rate"in d:ST.cfg["learning_rate"]=float(d["learning_rate"])
    ST._log("Config updated");return jsonify({"ok":True})
@app.route("/api/workers")
def api_w():return jsonify({"workers":ping_workers(ST.worker_urls,ST.token) if ST.worker_urls else[],"device":ST.device,"ips":get_local_ips(),"include_local":ST.include_local,"token":ST.token})
@app.route("/api/workers/add",methods=["POST"])
def wa():u=(request.json or{}).get("url","").strip().rstrip("/");u and u not in ST.worker_urls and ST.worker_urls.append(u);return jsonify({"ok":True})
@app.route("/api/workers/remove",methods=["POST"])
def wr():u=(request.json or{}).get("url","").strip().rstrip("/");ST.worker_urls=[w for w in ST.worker_urls if w!=u];return jsonify({"ok":True})
@app.route("/api/workers/ping",methods=["POST"])
def wp():return jsonify({"ok":True})
@app.route("/api/workers/scan",methods=["POST"])
def ws():
    found=[];[found.extend(scan_subnet(ip,DEFAULT_WORKER_PORT,ST.token)) for ip in get_local_ips()]
    for f in found:
        u=f.get("url","").rstrip("/")
        if u and u not in ST.worker_urls:ST.worker_urls.append(u)
    ST._log("Scan:"+str(len(found)));return jsonify({"ok":True})
@app.route("/api/reset",methods=["POST"])
def api_reset():ST.init_nets();ST.rh.clear();ST.cr=0;return jsonify({"ok":True})
@app.route("/api/dream",methods=["POST"])
def api_dream():
    info=REGIMES.dream(ST.cfg.get("dream_min_act",5),ST.cr);ST._log("DREAM: +"+str(info["births"])+" -"+str(info["prunes"])+" merged:"+str(info["merges"])+" -> "+str(info["after"])+" regimes: "+", ".join(info.get("names",[])))
    return jsonify({"ok":True,**info})
@app.route("/api/bx",methods=["POST"])
def api_bx():
    """Full debug snapshot — captures everything for bug reporting."""
    client_errors=(request.json or{}).get("js_errors",[])
    snap={"version":VERSION,"timestamp":now_ts(),"mode":"DEMO" if MOCK_MODE else "LIVE","app_mode":APP_MODE,"device":ST.device,
        "config":ST.cfg,"current_round":ST.cr,"is_training":ST.is_training,
        "replay_sizes":{"A":ST.replays["A"].size() if "A" in ST.replays else 0,"B":ST.replays["B"].size() if "B" in ST.replays else 0},
        "leaderboard":ST.lb.top()[:15],"round_history":ST.rh[-20:],
        "regimes":REGIMES.stats(),"worker_urls":ST.worker_urls,"worker_pings":ping_workers(ST.worker_urls,ST.token) if ST.worker_urls else[],
        "worker_state":{"jobs":WS.jobs_done,"games":WS.total_games,"busy":WS.busy,"started":WS.started_at},
        "event_log":list(ST.log),"error_log":list(ST.errors),"worker_log":list(WS.log),
        "client_js_errors":client_errors,"local_ips":get_local_ips(),"hostname":socket.gethostname(),"pid":os.getpid(),
        "python_version":sys.version,"mock_mode":MOCK_MODE}
    # Playground state
    if ST.pg_state:
        snap["playground"]={"board":ST.bj(ST.pg_state),"moves":ST.pg_moves,"move_count":ST.pg_state.move_count,
            "to_play":ST.pg_state.to_play,"game_over":ST.pg_state.game_over()}
        _,_,_,active=REGIMES.query(ST.pg_state.board_array(),ST.pg_state.to_play,ST.pg_state.move_count)
        snap["playground"]["active_regimes"]=active
    # Net info
    if not MOCK_MODE and "A" in ST.nets:
        try:snap["net_graph"]=ST.nets["A"].snapshot_graph();snap["net_params"]=sum(p.numel() for p in ST.nets["A"].parameters())
        except:pass
    ST._log("[BX] Debug snapshot captured")
    return jsonify(snap)
@app.route("/api/train/start",methods=["POST"])
def ts():
    if ST.is_training:return jsonify({"ok":False})
    if not ST.nets:ST.init_nets()
    ST.is_training=True;ST._stop=False;threading.Thread(target=_tl,daemon=True).start();return jsonify({"ok":True})
@app.route("/api/train/stop",methods=["POST"])
def tp():ST._stop=True;ST._log("Stop");return jsonify({"ok":True})
@app.route("/api/playground/new",methods=["POST"])
def pn():ST.pg_state=GoState.new();ST.pg_moves=[];return _pg_resp()
@app.route("/api/playground/move",methods=["POST"])
def pmv():
    if not ST.pg_state:ST.pg_state=GoState.new();ST.pg_moves=[]
    mi=PASS_MOVE if request.json.get("move")=="pass"else int(request.json.get("move"));ns=ST.pg_state.try_play(mi)
    if ns is None:return jsonify({"error":"illegal","board":ST.bj(ST.pg_state),"moves":ST.pg_moves})
    ST.pg_state=ns;ST.pg_moves.append(move_to_str(mi));return _pg_resp()
@app.route("/api/playground/ai",methods=["POST"])
def pai():
    if not ST.pg_state:ST.pg_state=GoState.new();ST.pg_moves=[]
    if ST.pg_state.game_over():return _pg_resp()
    if MOCK_MODE:mv=random.choice(ST.pg_state.legal_moves())
    else:
        net=ST.nets.get("A")
        if not net:ST.init_nets();net=ST.nets["A"]
        mv=eval_move(net,ST.device,ST.pg_state,sims=ST.cfg.get("selfplay_sims",20))
    ns=ST.pg_state.try_play(mv)
    if ns is None:ns=ST.pg_state.try_play(PASS_MOVE);mv=PASS_MOVE
    if ns:ST.pg_state=ns;ST.pg_moves.append(move_to_str(mv))
    return _pg_resp()
def _pg_resp():
    st=ST.pg_state;b=st.board_array()
    _,_,_,active=REGIMES.query(b,st.to_play,st.move_count)
    if MOCK_MODE:
        phase=st.move_count/MAX_GAME_LEN;_,_,conf,_=REGIMES.query(b,st.to_play,st.move_count)
        ew=[max(.05,.35-phase*.3),max(.05,.2+phase*.1),max(.05,.15+phase*.1),max(.05,.15+phase*.2),max(.05,conf*.3)];s=sum(ew);ew=[x/s for x in ew]
    else:
        try:_,_,aux=infer_aux(ST.nets["A"],ST.device,st);ew=aux["weights"]
        except:ew=None
    return jsonify({"board":ST.bj(st),"moves":ST.pg_moves,"experts":ew,"active_regimes":active})

# === TRAINING LOOP ===
def _tl():
    try:_mt() if MOCK_MODE else _rt()
    except Exception as e:ST._err("Training failed: "+str(e),e)
    finally:ST.is_training=False
def _mt():
    for ri in range(1,ST.cfg["rounds"]+1):
        if ST._stop:ST._log("Stopped");break
        ST.cr=ri;ST._log("=== Round "+str(ri)+" ===")
        for n in["A","B"]:
            ST._log("  SP "+n);results=push_parallel("mock",ST.cfg["selfplay_sims"],ST.cfg["selfplay_jobs_per_round"]*ST.cfg["games_per_job"],ST.worker_urls,ST.token,ST.include_local,ST.device,rn=ri)
            for res in results:ST._log("    "+res.get("worker_name","?")+" "+str(len(res.get("results",[])))+"g");ST.replays[n].add([None]*len(res.get("results",[])))
        la=max(.3,4.5-ri*.4+random.uniform(-.2,.2));lb_=max(.3,4.6-ri*.38+random.uniform(-.2,.2))
        phase=ri/ST.cfg["rounds"];_,_,kc,_=REGIMES.query(np.zeros((6,6),dtype=np.int8),1,int(phase*60))
        ew=[max(.05,.35-phase*.3),max(.05,.2+phase*.1),max(.05,.15+phase*.08),max(.05,.15+phase*.18),max(.05,kc*.2+.05)];s=sum(ew);ew=[w/s for w in ew]
        edges=[[random.uniform(.1,.5) for _ in range(5)] for _ in range(5)]
        for row in edges:s=sum(row);row[:]=[v/s for v in row]
        wa=random.randint(2,6);ST.lb.update("A_current","B_current",wa/8)
        ST.rh.append({"r":ri,"la":round(la,4),"lb":round(lb_,4),"avr":round(min(1,.3+ri*.08),3),"bvr":round(min(1,.3+ri*.07),3),"h2h":str(wa)+"/8","ew":ew,"graph":{"experts":list(EXPERT_NAMES),"edges":edges,"temp":1.0}})
        rs=REGIMES.stats();ST._log("  Loss A:"+str(round(la,3))+" B:"+str(round(lb_,3))+" Regimes:"+str(rs["num_regimes"])+" buf:"+str(rs["buffer_size"]))
        di=ST.cfg.get("dream_interval",5)
        if di>0 and ri%di==0:
            ST._log("  DREAMING...");info=REGIMES.dream(ST.cfg.get("dream_min_act",5),ri)
            names=", ".join(info.get("names",[]))[:80]
            ST._log("    +"+str(info["births"])+" -"+str(info["prunes"])+" m:"+str(info["merges"])+" -> "+str(info["after"])+" ["+names+"]")
        ST.lb.ensure("A_r"+str(ri).zfill(3),ST.lb.ratings.get("A_current",1000));ST.lb.ensure("B_r"+str(ri).zfill(3),ST.lb.ratings.get("B_current",1000))
    ST._log("Done")
def _rt():
    cfg=ST.cfg;total=cfg["selfplay_jobs_per_round"]*cfg["games_per_job"]
    for ri in range(1,cfg["rounds"]+1):
        if ST._stop:ST._log("Stopped");break
        ST.cr=ri;ST._log("=== Round "+str(ri)+" ===")
        for nn in["A","B"]:
            ST._log("SP "+nn);mb=net_to_b64(ST.nets[nn])
            for res in push_parallel(mb,cfg["selfplay_sims"],total,ST.worker_urls,ST.token,ST.include_local,ST.device,rn=ri):
                if res.get("samples"):
                    pk=decompress_obj(res["samples"]);ST.replays[nn].add([Sample(np.array(i[0],dtype=np.float32),np.array(i[1],dtype=np.float32),float(i[2])) for i in pk])
                ST._log("  "+nn+"."+res.get("worker_name","?")+" "+str(len(res.get("results",[])))+"g")
        ST._log("Train...");info={"r":ri}
        for n in["A","B"]:
            tr=train_team(ST.nets[n],ST.replays[n],ST.device,cfg["train_steps"],cfg["batch_size"],cfg["learning_rate"])
            info["l"+n.lower()]=tr.get("total_loss")
            if tr["total_loss"]:ST._log("  "+n+":"+str(round(tr["total_loss"],4))+" bal:"+str(round(tr.get("balance_loss",0),4)))
            if tr.get("graph"):info["graph"]=tr["graph"]
        ev=evaluate_pair(ST.nets,ST.device,cfg.get("eval_sims",28),cfg["eval_games"]);sa=ev["wins_A"]/max(1,ev["games"]);ST.lb.update("A_current","B_current",sa)
        info["avr"]=ev.get("A_vs_random");info["bvr"]=ev.get("B_vs_random");info["h2h"]=str(ev["wins_A"])+"/"+str(ev["games"])
        try:_,_,aux=infer_aux(ST.nets["A"],ST.device,GoState.new());info["ew"]=aux["weights"]
        except:pass
        ST.rh.append(info);rs=REGIMES.stats();ST._log("  A "+str(ev["wins_A"])+"/"+str(ev["games"])+" reg:"+str(rs["num_regimes"]))
        di=cfg.get("dream_interval",5)
        if di>0 and ri%di==0:
            ST._log("  DREAMING...");dinfo=REGIMES.dream(cfg.get("dream_min_act",5),ri)
            ST._log("    +"+str(dinfo["births"])+" -"+str(dinfo["prunes"])+" -> "+str(dinfo["after"])+" ["+", ".join(dinfo.get("names",[]))[:80]+"]")
        ST.lb.ensure("A_r"+str(ri).zfill(3),ST.lb.ratings.get("A_current",1000));ST.lb.ensure("B_r"+str(ri).zfill(3),ST.lb.ratings.get("B_current",1000))
    ST._log("Done")

# === CLI ===
def run_coord(port=DEFAULT_COORD_PORT,token="",device=None,inter=True):
    global APP_MODE;APP_MODE="coordinator"
    if device:ST.device=device
    ST.token=token;ST.init_nets()
    if inter:port=resolve_port(port)
    ips=get_local_ips();print("\n"+"="*52+"\n  EXOSFEAR v4 COORDINATOR\n  http://localhost:"+str(port))
    for ip in ips:print("  http://"+ip+":"+str(port))
    print("  "+("DEMO" if MOCK_MODE else "LIVE")+" | Regime Nets + Dreaming\n"+"="*52)
    app.config["AUTH_TOKEN"]=token;app.config["WORKER_PORT"]=port;app.run(host="0.0.0.0",port=port,debug=False,use_reloader=False)
def run_worker(port=DEFAULT_WORKER_PORT,token="",device=None,inter=True):
    global APP_MODE;APP_MODE="worker"
    if device:ST.device=device
    if inter:port=resolve_port(port)
    ips=get_local_ips();print("\n"+"="*52+"\n  EXOSFEAR v4 WORKER")
    for ip in ips:print("  -> http://"+ip+":"+str(port))
    print("  "+("DEMO" if MOCK_MODE else "LIVE")+"\n"+"="*52)
    WS._log("Started port "+str(port));app.config["AUTH_TOKEN"]=token;app.config["WORKER_PORT"]=port;app.run(host="0.0.0.0",port=port,debug=False,use_reloader=False)
def menu():
    print("\n  EXOSFEAR MicroGo KG v4\n  5-node graph + Regime Nets + Dreaming\n  1) Coordinator  2) Worker\n  "+("DEMO" if MOCK_MODE else "LIVE")+"\n")
    try:ch=input("  [1]: ").strip() or"1"
    except:ch="1"
    if ch in("1","c"):
        tk=secrets.token_urlsafe(8)
        try:r=input("  Token ["+tk+"]: ").strip();tk=r if r else tk
        except:pass
        run_coord(token=tk)
    elif ch in("2","w"):
        tk=""
        try:tk=input("  Token [none]: ").strip()
        except:pass
        run_worker(token=tk)
def main():
    p=argparse.ArgumentParser();p.add_argument("mode",nargs="?",default="menu",choices=["menu","coordinator","worker"])
    p.add_argument("--port",type=int);p.add_argument("--token",default="");p.add_argument("--device",default=None)
    a=p.parse_args()
    if a.mode=="coordinator":run_coord(port=a.port or DEFAULT_COORD_PORT,token=a.token,device=a.device,inter=a.port is None)
    elif a.mode=="worker":run_worker(port=a.port or DEFAULT_WORKER_PORT,token=a.token,device=a.device,inter=a.port is None)
    else:menu()
if __name__=="__main__":main()
