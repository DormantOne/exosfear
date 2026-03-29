#!/usr/bin/env python3
"""
EXOSFEAR MicroGo KG - All-in-one distributed training for 6x6 Go.
Single script: coordinator (full dashboard) or worker (mini dashboard + job server).

    pip install flask torch numpy       # full mode
    pip install flask numpy             # demo mode (mock engine)

    python microgo.py                   # interactive menu
    python microgo.py coordinator       # coordinator
    python microgo.py worker            # worker
"""
from __future__ import annotations
import argparse,base64,concurrent.futures,gzip,hashlib,io,ipaddress,json
import math,os,pickle,random,secrets,signal,socket,subprocess,sys
import threading,time
from collections import deque
from dataclasses import dataclass,field
from pathlib import Path
from typing import Any,Dict,List,Optional,Sequence,Tuple
import numpy as np

MOCK_MODE=False
try:
    import torch,torch.nn as nn,torch.nn.functional as F
    torch.set_num_threads(1)
    try:torch.set_num_interop_threads(1)
    except:pass
except Exception:
    MOCK_MODE=True

BOARD_SIZE=6;PASS_MOVE=36;ALL_MOVES=37;KOMI=3.5;MAX_GAME_LEN=120
SEED=42;EXPERT_NAMES=["opening","tactics","territory","endgame"]
VERSION="exosfear-microgo-kg-2.0"
DEFAULT_COORD_PORT=5000;DEFAULT_WORKER_PORT=8765

def now_ts():return time.strftime("%Y-%m-%d %H:%M:%S")
def choose_device():return "cuda" if not MOCK_MODE and torch.cuda.is_available() else "cpu"

# === NETWORK HELPERS ===
def get_local_ips():
    ips=[]
    def add(ip):
        ip=(ip or"").strip()
        if ip and not ip.startswith("127.") and":"not in ip and ip not in ips:ips.append(ip)
    try:
        for res in socket.getaddrinfo(socket.gethostname(),None,socket.AF_INET):add(res[4][0])
    except:pass
    for probe in["8.8.8.8","1.1.1.1","192.168.1.1","10.0.0.1"]:
        try:s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);s.connect((probe,80));add(s.getsockname()[0]);s.close()
        except:pass
    return ips

def port_is_free(host,port):
    try:
        with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);s.bind((host,port))
        return True
    except OSError:return False

def find_free_port(host="0.0.0.0",preferred=8765,span=200):
    for p in range(preferred,preferred+span):
        if port_is_free(host,p):return p
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as s:s.bind((host,0));return int(s.getsockname()[1])

def get_port_pid(port):
    for cmd in[["lsof","-ti",f":{port}"],["fuser",f"{port}/tcp"]]:
        try:
            out=subprocess.check_output(cmd,stderr=subprocess.DEVNULL).decode().strip()
            if out:
                pid=int(out.split()[0].split("/")[0])
                try:cmdline=subprocess.check_output(["ps","-p",str(pid),"-o","args="],stderr=subprocess.DEVNULL).decode().strip()
                except:cmdline="unknown"
                return pid,cmdline
        except:pass
    return None,None

def resolve_port(preferred,label="Service"):
    if port_is_free("0.0.0.0",preferred):return preferred
    pid,cmd=get_port_pid(preferred)
    print(f"\n  Warning: Port {preferred} is in use.")
    if pid:print(f"     PID {pid}: {cmd}")
    alt=find_free_port("0.0.0.0",preferred+1)
    try:choice=input(f"  [K]ill it / [U]se {alt} instead / [Q]uit? [K]: ").strip().lower() or"k"
    except(EOFError,KeyboardInterrupt):choice="u"
    if choice=="k"and pid:
        try:os.kill(pid,signal.SIGTERM);time.sleep(0.5);print(f"  Killed PID {pid}.")
        except Exception as e:print(f"  Kill failed: {e}");return alt
        if port_is_free("0.0.0.0",preferred):return preferred
        return alt
    elif choice=="q":sys.exit(0)
    return alt

def http_json(url,method="GET",payload=None,token="",timeout=300):
    import urllib.request
    data=json.dumps(payload).encode() if payload is not None else None
    req=urllib.request.Request(url=url,data=data,method=method)
    req.add_header("Content-Type","application/json; charset=utf-8")
    if token:req.add_header("X-Token",token)
    with urllib.request.urlopen(req,timeout=timeout) as resp:return json.loads(resp.read().decode())

def scan_subnet(base_ip,port,token="",timeout=1.0):
    found=[]
    try:net=ipaddress.ip_network(base_ip+"/24",strict=False)
    except:return found
    candidates=[str(ip) for ip in net.hosts()]
    def check(ip):
        url=f"http://{ip}:{port}"
        try:
            data=http_json(url+"/health",method="GET",token=token,timeout=max(1,int(timeout)))
            if data.get("ok") and data.get("role")=="microgo_worker":data["url"]=url;return data
        except:pass
    with concurrent.futures.ThreadPoolExecutor(max_workers=48) as ex:
        futs={ex.submit(check,ip):ip for ip in candidates}
        for fut in concurrent.futures.as_completed(futs):
            try:
                res=fut.result()
                if res:found.append(res)
            except:pass
    found.sort(key=lambda x:x.get("url",""));return found

def ping_workers(urls,token):
    out=[]
    for url in urls:
        try:data=http_json(url.rstrip("/")+"/health",token=token,timeout=5);out.append({"url":url,**data})
        except Exception as e:out.append({"url":url,"ok":False,"error":str(e)[:80]})
    return out

def compress_obj(obj):return base64.b64encode(gzip.compress(pickle.dumps(obj,protocol=pickle.HIGHEST_PROTOCOL))).decode("ascii")
def decompress_obj(s):return pickle.loads(gzip.decompress(base64.b64decode(s.encode("ascii"))))

def move_to_str(m):
    if m==PASS_MOVE:return"pass"
    r,c=divmod(m,BOARD_SIZE);return f"{chr(65+c)}{r+1}"

# === GO ENGINE ===
class GoState:
    __slots__=("board_data","to_play","passes","move_count","_history")
    def __init__(self,board=None,to_play=1,passes=0,move_count=0,history=None):
        self.board_data=board if board is not None else np.zeros((BOARD_SIZE,BOARD_SIZE),dtype=np.int8)
        self.to_play=to_play;self.passes=passes;self.move_count=move_count
        self._history=history if history is not None else frozenset()
    @staticmethod
    def new():s=GoState();s._history=frozenset([s.board_data.tobytes()]);return s
    def board_array(self):return self.board_data.copy()
    def _nbrs(self,r,c):
        o=[]
        if r>0:o.append((r-1,c))
        if r+1<BOARD_SIZE:o.append((r+1,c))
        if c>0:o.append((r,c-1))
        if c+1<BOARD_SIZE:o.append((r,c+1))
        return o
    def _group_libs(self,board,r,c):
        color=int(board[r,c]);stack=[(r,c)];seen={(r,c)};group=[];libs=set()
        while stack:
            rr,cc=stack.pop();group.append((rr,cc))
            for nr,nc in self._nbrs(rr,cc):
                v=int(board[nr,nc])
                if v==0:libs.add((nr,nc))
                elif v==color and(nr,nc)not in seen:seen.add((nr,nc));stack.append((nr,nc))
        return group,libs
    def legal_moves(self):
        moves=[];board=self.board_array()
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if board[r,c]==0 and self.try_play(r*BOARD_SIZE+c)is not None:moves.append(r*BOARD_SIZE+c)
        moves.append(PASS_MOVE);return moves
    def try_play(self,move):
        board=self.board_array()
        if move==PASS_MOVE:return GoState(board,-self.to_play,self.passes+1,self.move_count+1,self._history)
        r,c=divmod(move,BOARD_SIZE)
        if board[r,c]!=0:return None
        board[r,c]=self.to_play
        for nr,nc in self._nbrs(r,c):
            if board[nr,nc]==-self.to_play:
                grp,libs=self._group_libs(board,nr,nc)
                if not libs:
                    for gr,gc in grp:board[gr,gc]=0
        grp,libs=self._group_libs(board,r,c)
        if not libs:return None
        bh=board.tobytes()
        if bh in self._history:return None
        return GoState(board,-self.to_play,0,self.move_count+1,self._history|{bh})
    def game_over(self):return self.passes>=2 or self.move_count>=MAX_GAME_LEN
    def final_score_black(self):
        board=self.board_array();sb=int(np.sum(board==1));sw=int(np.sum(board==-1))
        visited=np.zeros_like(board,dtype=np.uint8);tb=tw=0
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if board[r,c]!=0 or visited[r,c]:continue
                stack=[(r,c)];region=[];borders=set();visited[r,c]=1
                while stack:
                    rr,cc=stack.pop();region.append((rr,cc))
                    for nr,nc in self._nbrs(rr,cc):
                        v=int(board[nr,nc])
                        if v==0 and not visited[nr,nc]:visited[nr,nc]=1;stack.append((nr,nc))
                        elif v!=0:borders.add(v)
                if borders=={1}:tb+=len(region)
                elif borders=={-1}:tw+=len(region)
        return(sb+tb)-(sw+tw+KOMI)
    def winner(self):return 1 if self.final_score_black()>0 else-1

def encode_state(state):
    board=state.board_array();own=(board==state.to_play).astype(np.float32)
    opp=(board==-state.to_play).astype(np.float32);turn=np.full_like(own,1.0 if state.to_play==1 else 0.0)
    legal=np.zeros_like(own)
    for mv in state.legal_moves():
        if mv!=PASS_MOVE:r,c=divmod(mv,BOARD_SIZE);legal[r,c]=1.0
    age=np.full_like(own,min(state.move_count/MAX_GAME_LEN,1.0))
    return np.stack([own,opp,turn,legal,age],axis=0)

# === NEURAL NET ===
if not MOCK_MODE:
    class ResBlock(nn.Module):
        def __init__(self,ch):
            super().__init__();self.c1=nn.Conv2d(ch,ch,3,padding=1);self.b1=nn.BatchNorm2d(ch)
            self.c2=nn.Conv2d(ch,ch,3,padding=1);self.b2=nn.BatchNorm2d(ch)
        def forward(self,x):return F.relu(x+self.b2(self.c2(F.relu(self.b1(self.c1(x))))))
    class ExpertTower(nn.Module):
        def __init__(self,ch,dd,blocks=1):
            super().__init__();self.blocks=nn.Sequential(*[ResBlock(ch) for _ in range(blocks)])
            self.ph=nn.Sequential(nn.Conv2d(ch,2,1),nn.BatchNorm2d(2),nn.ReLU())
            self.pf=nn.Linear(2*BOARD_SIZE*BOARD_SIZE,ALL_MOVES)
            self.vh=nn.Sequential(nn.Conv2d(ch,1,1),nn.BatchNorm2d(1),nn.ReLU())
            self.vf1=nn.Linear(BOARD_SIZE*BOARD_SIZE,48);self.vf2=nn.Linear(48,1);self.df=nn.Linear(ch,dd)
        def forward(self,base):
            h=self.blocks(base);logits=self.pf(self.ph(h).flatten(1))
            v=torch.tanh(self.vf2(F.relu(self.vf1(self.vh(h).flatten(1))))).squeeze(-1)
            return logits,v,torch.tanh(self.df(F.adaptive_avg_pool2d(h,1).flatten(1)))
    class GraphTeamNet(nn.Module):
        def __init__(self,ch=24,sb=1,eb=1,dd=32):
            super().__init__();ne=len(EXPERT_NAMES);self.expert_names=list(EXPERT_NAMES)
            self.stem=nn.Sequential(nn.Conv2d(5,ch,3,padding=1),nn.BatchNorm2d(ch),nn.ReLU())
            self.shared=nn.Sequential(*[ResBlock(ch) for _ in range(sb)])
            self.experts=nn.ModuleList([ExpertTower(ch,dd,eb) for _ in range(ne)])
            self.expert_token=nn.Parameter(torch.randn(ne,dd)*0.05)
            self.router=nn.Sequential(nn.Linear(ch+3,64),nn.ReLU(),nn.Linear(64,ne))
            self.edge_logits=nn.Parameter(torch.zeros(ne,ne))
            self.conf_head=nn.Linear(dd,1);self.router_temp=nn.Parameter(torch.tensor(1.0))
        def graph_matrix(self):return torch.softmax(self.edge_logits,dim=-1)
        def forward(self,x,return_aux=False):
            ne=len(self.expert_names);base=self.shared(self.stem(x))
            pooled=F.adaptive_avg_pool2d(base,1).flatten(1)
            rl=self.router(torch.cat([pooled,x[:,4].mean(dim=(1,2)).unsqueeze(-1),x[:,3].mean(dim=(1,2)).unsqueeze(-1),x[:,2].mean(dim=(1,2)).unsqueeze(-1)],-1))
            pols,vals,descs=[],[],[]
            for i,exp in enumerate(self.experts):
                lo,va,de=exp(base);de=de+self.expert_token[i].unsqueeze(0);pols.append(lo);vals.append(va);descs.append(de)
            ps=torch.stack(pols,1);vs=torch.stack(vals,1);ds=torch.stack(descs,1)
            edges=self.graph_matrix();md=torch.einsum("ij,bjd->bid",edges,ds)
            conf=self.conf_head(torch.tanh(ds+md)).squeeze(-1)
            temp=torch.clamp(self.router_temp.abs(),0.3,3.0);w=torch.softmax((rl+conf)/temp,dim=-1)
            fp=(ps*w.unsqueeze(-1)).sum(1);fv=(vs*w).sum(1)
            if not return_aux:return fp,fv
            return fp,fv,{"weights":w,"conf":conf}
        def snapshot_graph(self):
            with torch.no_grad():
                return{"experts":list(self.expert_names),"edges":self.graph_matrix().cpu().numpy().tolist(),"temp":float(torch.clamp(self.router_temp.abs(),0.3,3.0).cpu())}
    def new_team(device):net=GraphTeamNet();net.to(device);net.eval();return net
    def net_to_b64(net):bio=io.BytesIO();torch.save(net.state_dict(),bio);return base64.b64encode(gzip.compress(bio.getvalue())).decode("ascii")
    def net_from_b64(payload,device):
        net=new_team(device);net.load_state_dict(torch.load(io.BytesIO(gzip.decompress(base64.b64decode(payload.encode("ascii")))),map_location=device,weights_only=False));net.eval();return net
    def infer_aux(net,device,state):
        x=torch.from_numpy(encode_state(state)).unsqueeze(0).to(device)
        with torch.no_grad():lo,va,aux=net(x,return_aux=True)
        return(lo.squeeze(0).cpu().numpy(),float(va.squeeze(0).cpu()),{"weights":aux["weights"].squeeze(0).cpu().numpy().tolist(),"conf":aux["conf"].squeeze(0).cpu().numpy().tolist()})

# === MCTS ===
@dataclass
class TreeNode:
    prior:float;to_play:int;visit_count:int=0;value_sum:float=0.0
    children:Dict[int,"TreeNode"]=field(default_factory=dict);expanded:bool=False
    def value(self):return 0.0 if self.visit_count==0 else self.value_sum/self.visit_count

if not MOCK_MODE:
    class MCTS:
        def __init__(self,net,device,sims=32,c_puct=1.5):self.net=net;self.device=device;self.sims=sims;self.c=c_puct
        def _eval(self,state):
            x=torch.from_numpy(encode_state(state)).unsqueeze(0).to(self.device)
            with torch.no_grad():lo,va=self.net(x);lo=lo.squeeze(0).cpu().numpy();val=float(va.squeeze(0).cpu())
            legal=state.legal_moves();mask=np.zeros(ALL_MOVES,dtype=np.float32);mask[legal]=1.0
            lo[mask==0]=-1e9;pr=np.exp(lo-lo.max());pr*=mask;s=pr.sum()
            return(mask/max(mask.sum(),1.0) if s<=0 else pr/s),val
        def _expand(self,node,state,noise=False):
            priors,value=self._eval(state);legal=state.legal_moves()
            if noise and legal:
                n=np.random.dirichlet([0.3]*len(legal))
                for i,mv in enumerate(legal):priors[mv]=0.75*priors[mv]+0.25*n[i]
            for mv in legal:
                cs=state.try_play(mv)
                if cs:node.children[mv]=TreeNode(float(priors[mv]),cs.to_play)
            node.expanded=True;return value
        def _select(self,node):
            tv=math.sqrt(max(1,node.visit_count));best_s=-1e9;best_m=PASS_MOVE;best_c=None
            for mv,ch in node.children.items():
                sc=-ch.value()+self.c*ch.prior*tv/(1+ch.visit_count)
                if sc>best_s:best_s=sc;best_m=mv;best_c=ch
            return best_m,best_c
        def run(self,root_state):
            root=TreeNode(1.0,root_state.to_play)
            if root_state.game_over():v=np.zeros(ALL_MOVES,dtype=np.float32);v[PASS_MOVE]=1.0;return v
            self._expand(root,root_state,noise=True)
            for _ in range(self.sims):
                node=root;state=root_state;path=[node]
                while node.expanded and node.children:
                    mv,child=self._select(node);ns=state.try_play(mv)
                    if not ns:break
                    node=child;state=ns;path.append(node)
                    if state.game_over():break
                value=(1.0 if state.winner()==state.to_play else-1.0) if state.game_over() else self._expand(node,state)
                for bn in reversed(path):bn.visit_count+=1;bn.value_sum+=value;value=-value
            visits=np.zeros(ALL_MOVES,dtype=np.float32)
            for mv,ch in root.children.items():visits[mv]=ch.visit_count
            return visits
    def sample_move(visits,state,temp):
        legal=state.legal_moves();pr=np.zeros_like(visits);pr[legal]=visits[legal]
        if pr.sum()<=0:pr[legal]=1.0
        if temp<=1e-4:mv=int(np.argmax(pr));oh=np.zeros_like(pr);oh[mv]=1.0;return mv,oh
        pp=pr**(1.0/temp);s=pp.sum()
        if s<=0:pp[legal]=1.0;s=pp.sum()
        pp/=s;return int(np.random.choice(np.arange(ALL_MOVES),p=pp)),pp
    def eval_move(net,device,state,sims):
        mcts=MCTS(net,device,sims,1.3);visits=mcts.run(state)
        legal=state.legal_moves();masked=np.zeros_like(visits);masked[legal]=visits[legal]
        return PASS_MOVE if masked.sum()<=0 else int(np.argmax(masked))

# === SELF-PLAY / TRAINING ===
@dataclass
class Sample:
    state_planes:np.ndarray;policy:np.ndarray;z:float

class ReplayBuffer:
    def __init__(self,cap=50000):self.cap=cap;self.data=[];self.lock=threading.Lock()
    def add(self,batch):
        with self.lock:self.data.extend(batch);self.data=self.data[-self.cap:] if len(self.data)>self.cap else self.data
    def size(self):
        with self.lock:return len(self.data)
    def sample_batch(self,bs):
        with self.lock:
            if len(self.data)<bs:return None
            idx=np.random.choice(len(self.data),bs,replace=False);ch=[self.data[i] for i in idx]
        return(np.stack([s.state_planes for s in ch]),np.stack([s.policy for s in ch]),np.array([s.z for s in ch],dtype=np.float32))

if not MOCK_MODE:
    def self_play_game(net,device,sims,temp_moves=10):
        state=GoState.new();samples=[];moves=[];gate_trace=[];mcts=MCTS(net,device,sims)
        while not state.game_over():
            _,_,aux=infer_aux(net,device,state);gate_trace.append(aux["weights"])
            visits=mcts.run(state);temp=1.0 if state.move_count<temp_moves else 1e-6
            mv,policy=sample_move(visits,state,temp)
            samples.append((encode_state(state),policy.astype(np.float32),state.to_play))
            moves.append(move_to_str(mv));ns=state.try_play(mv)
            if not ns:ns=state.try_play(PASS_MOVE)
            if not ns:break
            state=ns
        winner=state.winner()
        return{"winner":winner,"score_black":state.final_score_black(),"num_moves":len(moves),"moves":moves,
               "samples":[Sample(s,p,1.0 if pl==winner else-1.0) for s,p,pl in samples],
               "avg_gate":np.mean(gate_trace,axis=0).tolist() if gate_trace else[0.0]*4}
    def do_selfplay_job(model_b64,device,sims,games):
        net=net_from_b64(model_b64,device);results=[];packed=[];gm=[]
        for _ in range(games):
            res=self_play_game(net,device,sims);results.append({k:v for k,v in res.items() if k!="samples"})
            gm.append(res.get("avg_gate",[0.25]*4))
            for s in res["samples"]:packed.append([s.state_planes.tolist(),s.policy.tolist(),float(s.z)])
        return{"results":results,"samples":compress_obj(packed),"avg_gate":np.mean(gm,axis=0).tolist() if gm else[0.25]*4}
    def train_team(net,replay,device,steps=100,bs=64,lr=1e-3):
        if replay.size()<bs:return{"steps":0,"total_loss":None}
        net.train();opt=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=1e-4);pl,vl,tl=[],[],[]
        for _ in range(steps):
            b=replay.sample_batch(bs)
            if not b:break
            x,p_tgt,z=[torch.from_numpy(a).to(device) for a in b];logits,value=net(x)
            ploss=-(p_tgt*F.log_softmax(logits,-1)).sum(-1).mean();vloss=F.mse_loss(value,z);loss=ploss+vloss
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.0);opt.step()
            pl.append(float(ploss.cpu()));vl.append(float(vloss.cpu()));tl.append(float(loss.cpu()))
        net.eval()
        return{"steps":len(tl),"policy_loss":float(np.mean(pl)) if pl else None,"value_loss":float(np.mean(vl)) if vl else None,"total_loss":float(np.mean(tl)) if tl else None,"graph":net.snapshot_graph()}
    def evaluate_pair(nets,device,sims,games):
        def play_match(nb,nw):
            state=GoState.new()
            while not state.game_over():
                mv=eval_move(nb if state.to_play==1 else nw,device,state,sims);ns=state.try_play(mv)
                if not ns:ns=state.try_play(PASS_MOVE)
                if not ns:break
                state=ns
            return state.winner()
        wins_a=0;half=max(1,games//2)
        for _ in range(half):
            if play_match(nets["A"],nets["B"])==1:wins_a+=1
        for _ in range(games-half):
            if play_match(nets["B"],nets["A"])==-1:wins_a+=1
        def vs_rand(net,n=4):
            w=0
            for _ in range(n):
                state=GoState.new()
                while not state.game_over():
                    mv=eval_move(net,device,state,sims) if state.to_play==1 else random.choice(state.legal_moves())
                    ns=state.try_play(mv)
                    if not ns:ns=state.try_play(PASS_MOVE)
                    if not ns:break
                    state=ns
                if state.winner()==1:w+=1
            return round(w/max(1,n),3)
        return{"games":games,"wins_A":wins_a,"wins_B":games-wins_a,"A_vs_random":vs_rand(nets["A"],max(4,games//2)),"B_vs_random":vs_rand(nets["B"],max(4,games//2))}

class Leaderboard:
    def __init__(self):self.ratings={}
    def ensure(self,name,r=1000.0):self.ratings.setdefault(name,r)
    def update(self,a,b,sa,k=24.0):
        self.ensure(a);self.ensure(b);ea=1.0/(1.0+10**((self.ratings[b]-self.ratings[a])/400.0))
        self.ratings[a]+=k*(sa-ea);self.ratings[b]+=k*((1-sa)-(1-ea))
    def top(self):return sorted(self.ratings.items(),key=lambda x:x[1],reverse=True)

# === PARALLEL DISPATCH ===
def push_selfplay_parallel(model_b64,sims,total_games,worker_urls,token,include_local,device,timeout=600):
    workers=(["__local__"] if include_local else[])+list(worker_urls)
    if not workers:workers=["__local__"]
    games_per=max(1,total_games//len(workers));remainder=total_games-games_per*len(workers)
    assignments=[(workers[i],games_per+(1 if i<remainder else 0)) for i in range(len(workers))]
    assignments=[(w,g) for w,g in assignments if g>0]
    results=[]
    def run_one(worker,num_games):
        t0=time.time()
        if worker=="__local__":
            if MOCK_MODE:
                time.sleep(0.3*num_games)
                return{"ok":True,"results":[{"winner":random.choice([1,-1])}]*num_games,"samples":compress_obj([]),"avg_gate":[0.25]*4,"worker_name":"local","seconds":round(time.time()-t0,2)}
            res=do_selfplay_job(model_b64,device,sims,num_games);res["worker_name"]="local";res["seconds"]=round(time.time()-t0,2);return res
        else:
            try:return http_json(worker.rstrip("/")+"/selfplay",method="POST",payload={"model":model_b64,"sims":sims,"games":num_games},token=token,timeout=timeout)
            except Exception as e:
                if MOCK_MODE:return{"ok":True,"results":[{"winner":1}]*num_games,"samples":compress_obj([]),"avg_gate":[0.25]*4,"worker_name":"fallback("+worker+")","seconds":round(time.time()-t0,2)}
                res=do_selfplay_job(model_b64,device,sims,num_games);res["worker_name"]="fallback("+worker+")";res["seconds"]=round(time.time()-t0,2);return res
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1,len(assignments))) as ex:
        futs={ex.submit(run_one,w,g):(w,g) for w,g in assignments}
        for fut in concurrent.futures.as_completed(futs):
            try:results.append(fut.result())
            except Exception as e:results.append({"ok":False,"error":str(e),"worker_name":"error","results":[],"samples":compress_obj([]),"avg_gate":[0.25]*4,"seconds":0})
    return results

# === APP STATE ===
class WorkerState:
    def __init__(self):
        self.jobs_done=0;self.total_games=0;self.total_samples=0;self.busy=False
        self.last_job_time=None;self.log=deque(maxlen=200);self.started_at=now_ts()
    def _log(self,msg):self.log.append({"ts":now_ts(),"msg":msg})
WORKER_STATE=WorkerState()

class AppState:
    def __init__(self):
        self.device=choose_device();self.nets={};self.replays={};self.leaderboard=Leaderboard()
        self.worker_urls=[];self.token="";self.include_local=True
        self.cfg={"rounds":6,"selfplay_sims":20,"eval_sims":28,"games_per_job":2,"selfplay_jobs_per_round":2,"train_steps":60,"batch_size":64,"learning_rate":0.001,"eval_games":8}
        self.log=deque(maxlen=500);self.round_history=[];self.current_round=0;self.is_training=False;self._stop_flag=False
        self.playground_state=None;self.playground_moves=[]
    def _log(self,msg):self.log.append({"ts":now_ts(),"msg":msg})
    def init_nets(self):
        for n in["A","B"]:
            self.nets[n]=new_team(self.device) if not MOCK_MODE else "mock_"+n
            self.replays[n]=ReplayBuffer();self.leaderboard.ensure(n+"_current")
        self._log("Nets initialized"+(" (demo)" if MOCK_MODE else ""))
    def get_board_json(self,state):
        board=state.board_array();cells=[];legal_rc=[]
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):cells.append({"r":r,"c":c,"v":int(board[r,c])})
        for mv in state.legal_moves():
            if mv==PASS_MOVE:legal_rc.append({"r":-1,"c":-1,"move":mv,"label":"pass"})
            else:rr,cc=divmod(mv,BOARD_SIZE);legal_rc.append({"r":rr,"c":cc,"move":mv,"label":move_to_str(mv)})
        return{"size":BOARD_SIZE,"cells":cells,"to_play":state.to_play,"to_play_label":"Black" if state.to_play==1 else "White",
               "passes":state.passes,"move_count":state.move_count,"game_over":state.game_over(),"legal_moves":legal_rc,
               "score_black":state.final_score_black() if state.game_over() else None}
ST=AppState();APP_MODE="coordinator"

# === FLASK ===
from flask import Flask,Response,request,jsonify
app=Flask(__name__);app.secret_key=secrets.token_hex(16)

# === CSS (shared) ===
CSS="""
:root{--bg:#0b0d0e;--bg2:#121618;--bg3:#1a1e22;--brd:#262c32;--fg:#c5cad0;--fg2:#8a9099;--fg3:#585e66;--gold:#c9a227;--gold2:#e0be4a;--teal:#3aafa9;--rose:#d45d5d;--sky:#5b8fd4;--stone-b:#181818;--stone-w:#e6e2d8}
*{margin:0;padding:0;box-sizing:border-box}body{background:var(--bg);color:var(--fg);font-family:'IBM Plex Mono',monospace;font-size:13px;line-height:1.55}::selection{background:var(--gold);color:var(--bg)}
.top{position:sticky;top:0;z-index:100;background:var(--bg);border-bottom:1px solid var(--brd);display:flex;align-items:center;height:50px;padding:0 1.25rem;gap:1.5rem}
.top h1{font-family:'Playfair Display',serif;font-size:1.15rem;font-weight:700;color:var(--gold);white-space:nowrap}.top h1 em{font-style:normal;color:var(--fg3);font-family:'IBM Plex Mono',monospace;font-size:.7rem;font-weight:400;margin-left:.5rem}
.tabs{display:flex}.tabs button{background:0;border:0;color:var(--fg3);font:inherit;font-size:11.5px;font-weight:500;padding:.55rem 1rem;cursor:pointer;border-bottom:2px solid transparent}.tabs button:hover{color:var(--fg)}.tabs button.on{color:var(--gold);border-bottom-color:var(--gold)}
.top-r{margin-left:auto;display:flex;align-items:center;gap:.7rem}
.dot{width:7px;height:7px;border-radius:50%;background:var(--fg3);display:inline-block}.dot.ok{background:var(--teal)}.dot.run{background:var(--gold);animation:pls 1.1s infinite}@keyframes pls{0%,100%{opacity:1}50%{opacity:.3}}
.pan{display:none;padding:1.25rem}.pan.on{display:block}.g2{display:grid;grid-template-columns:1fr 1fr;gap:1.1rem}@media(max-width:900px){.g2{grid-template-columns:1fr}}
.c{background:var(--bg2);border:1px solid var(--brd);border-radius:5px;overflow:hidden;margin-bottom:1.1rem}.c:last-child{margin-bottom:0}
.ch{padding:.55rem .9rem;border-bottom:1px solid var(--brd);font-size:10.5px;text-transform:uppercase;letter-spacing:.09em;color:var(--fg3)}.cb{padding:.85rem .9rem}
.fld{display:flex;flex-direction:column;gap:2px;font-size:11px;color:var(--fg2);margin-bottom:.6rem}
.fld input,.fld select{background:var(--bg3);border:1px solid var(--brd);color:var(--fg);font:inherit;font-size:12.5px;padding:5px 7px;border-radius:3px;outline:0}.fld input:focus{border-color:var(--gold)}
.btn{display:inline-flex;align-items:center;gap:.35rem;background:var(--gold);color:var(--bg);font:inherit;font-size:11.5px;font-weight:600;padding:6px 13px;border:0;border-radius:3px;cursor:pointer}.btn:hover{background:var(--gold2)}.btn:disabled{opacity:.35;cursor:not-allowed}
.btn.o{background:0;color:var(--gold);border:1px solid var(--gold)}.btn.o:hover{background:var(--gold);color:var(--bg)}.btn.d{background:var(--rose);color:#fff}.btn.s{font-size:10.5px;padding:4px 9px}
.mt{width:100%;border-collapse:collapse}.mt td{padding:4px 7px;border-bottom:1px solid var(--brd);font-size:12px}.mt td:first-child{color:var(--fg3);width:44%}.mt td:last-child{font-weight:500}
.eb{display:flex;align-items:center;gap:.45rem;margin:3px 0}.eb-l{width:66px;font-size:10.5px;color:var(--fg2);text-align:right}.eb-t{flex:1;height:15px;background:var(--bg3);border-radius:2px;overflow:hidden}.eb-f{height:100%;border-radius:2px;transition:width .35s}.eb-v{width:38px;font-size:10.5px;color:var(--fg3)}
.ag{display:inline-grid;gap:2px;font-size:10px}.ac{width:46px;height:24px;display:flex;align-items:center;justify-content:center;border-radius:2px;font-weight:500}.ah{color:var(--fg3);font-size:9.5px;text-align:center}
.la{background:var(--bg);border:1px solid var(--brd);border-radius:3px;font-size:11px;height:250px;overflow-y:auto;padding:.4rem;color:var(--fg2)}.la .ts{color:var(--fg3);margin-right:.4rem}
.lr{display:flex;align-items:center;gap:.5rem;padding:4px 0;border-bottom:1px solid var(--brd);font-size:12px}.lr-k{width:22px;color:var(--fg3);text-align:center}.lr-n{flex:1}.lr-e{font-weight:500;color:var(--gold);width:56px;text-align:right}
.wr{display:flex;align-items:center;gap:.5rem;padding:5px 0;border-bottom:1px solid var(--brd);font-size:12px}.wr .u{flex:1}.ws{font-size:11px}.ws.ok{color:var(--teal)}.ws.er{color:var(--rose)}
.ml{display:flex;flex-wrap:wrap;gap:3px}.mc{background:var(--bg3);border:1px solid var(--brd);border-radius:3px;padding:1px 5px;font-size:10.5px}.mc.b{border-left:3px solid var(--stone-b)}.mc.w{border-left:3px solid var(--stone-w)}
.em{color:var(--fg3);font-style:italic;font-size:12px;padding:.8rem 0}
.badge{display:inline-block;font-size:9.5px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;padding:2px 7px;border-radius:2px}
.badge.demo{background:rgba(212,93,93,.12);color:var(--rose)}.badge.live{background:rgba(58,175,169,.12);color:var(--teal)}.badge.work{background:rgba(91,143,212,.12);color:var(--sky)}
.big-url{font-size:1.1rem;color:var(--gold);background:var(--bg3);border:1px dashed var(--gold);border-radius:4px;padding:.6rem 1rem;text-align:center;margin:.6rem 0;cursor:pointer;user-select:all;word-break:break-all}
.step{display:flex;gap:.6rem;margin:.5rem 0}.step-n{background:var(--gold);color:var(--bg);width:22px;height:22px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;flex-shrink:0}.step-t{font-size:12px;line-height:1.5}
"""

FONT_LINK='<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=Playfair+Display:wght@400;700&display=swap" rel="stylesheet"/>'

def build_page(title, badge_class, badge_text, tabs_html, body_html, script):
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{title}</title>{FONT_LINK}<style>{CSS}</style></head><body>
<div class="top"><h1>EXOSFEAR <em>MicroGo KG</em></h1>{tabs_html}
<div class="top-r"><span class="badge {badge_class}">{badge_text}</span><span class="dot" id="sd"></span><span id="sl" style="font-size:11px;color:var(--fg3)">idle</span></div></div>
{body_html}<script>{script}</script></body></html>"""

# === WORKER PAGE ===
WORKER_BODY="""
<div style="padding:1.25rem"><div class="g2"><div>
<div class="c"><div class="ch">Connection Info</div><div class="cb">
<p style="font-size:12px;color:var(--fg2);margin-bottom:.4rem">Share this URL with the coordinator:</p>
<div class="big-url" id="wurl" onclick="navigator.clipboard.writeText(this.textContent)">loading...</div>
<p style="font-size:11px;color:var(--fg3)">Click to copy. Token: <span id="wtok">-</span></p></div></div>
<div class="c"><div class="ch">Setup</div><div class="cb">
<div class="step"><span class="step-n">1</span><span class="step-t">This worker is running. Keep this terminal open.</span></div>
<div class="step"><span class="step-n">2</span><span class="step-t">On your main machine: <code style="color:var(--gold)">python microgo.py coordinator</code></span></div>
<div class="step"><span class="step-n">3</span><span class="step-t">In the coordinator Workers tab, add the URL above or use Scan LAN.</span></div>
<div class="step"><span class="step-n">4</span><span class="step-t">Start training from the coordinator. Jobs will appear here.</span></div>
</div></div></div><div>
<div class="c"><div class="ch">Status</div><div class="cb"><table class="mt">
<tr><td>Status</td><td id="ws">idle</td></tr><tr><td>Device</td><td id="wdev">-</td></tr>
<tr><td>Jobs Done</td><td id="wjobs">0</td></tr><tr><td>Total Games</td><td id="wgames">0</td></tr>
<tr><td>Total Samples</td><td id="wsamp">0</td></tr><tr><td>Last Job</td><td id="wlast">-</td></tr>
<tr><td>Up Since</td><td id="wup">-</td></tr></table></div></div>
<div class="c"><div class="ch">Log</div><div class="cb" style="padding:0"><div class="la" id="wlog" style="height:300px"></div></div></div>
</div></div></div>"""

WORKER_JS="""
const $=id=>document.getElementById(id);
const api=async(p,b)=>{const o=b!==undefined?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)}:{};return(await fetch(p,o)).json()};
async function wpoll(){try{const d=await api('/api/worker/status');
$('wurl').textContent=d.url||'-';$('wtok').textContent=d.token?d.token.slice(-4):'-';
$('wdev').textContent=d.device;$('wjobs').textContent=d.jobs_done;$('wgames').textContent=d.total_games;
$('wsamp').textContent=d.total_samples;$('wlast').textContent=d.last_job_time||'-';$('wup').textContent=d.started_at;
$('ws').textContent=d.busy?'WORKING':'idle';$('ws').style.color=d.busy?'var(--gold)':'var(--teal)';
$('sd').className='dot '+(d.busy?'run':'ok');$('sl').textContent=d.busy?'working':'idle';
if(d.log&&d.log.length){const el=$('wlog');let h='';for(const e of d.log)h+='<div><span class="ts">'+e.ts+'</span>'+e.msg+'</div>';el.innerHTML=h;el.scrollTop=el.scrollHeight}
}catch(e){}}
wpoll();setInterval(wpoll,2000);
"""

def worker_page():
    return build_page("EXOSFEAR Worker","work","WORKER","",WORKER_BODY,WORKER_JS)

# === COORDINATOR PAGE ===
def coord_page():
    badge="demo" if MOCK_MODE else "live"
    tabs='<div class="tabs"><button class="on" data-tab="dash">Dashboard</button><button data-tab="play">Playground</button><button data-tab="work">Workers</button><button data-tab="conf">Config</button></div>'
    body="""
<div class="pan on" id="t-dash"><div class="g2"><div>
<div class="c"><div class="ch">Training</div><div class="cb"><table class="mt">
<tr><td>Round</td><td id="mr">0 / -</td></tr><tr><td>Replay A</td><td id="mra">0</td></tr><tr><td>Replay B</td><td id="mrb">0</td></tr>
<tr><td>Loss A</td><td id="mla">-</td></tr><tr><td>Loss B</td><td id="mlb">-</td></tr>
<tr><td>A vs Random</td><td id="mar">-</td></tr><tr><td>B vs Random</td><td id="mbr">-</td></tr><tr><td>Head-to-Head</td><td id="mhh">-</td></tr></table>
<div style="margin-top:.65rem;display:flex;gap:.4rem"><button class="btn" id="bs1" onclick="startT()">Start</button><button class="btn o" id="bs2" onclick="stopT()" disabled>Stop</button></div></div></div>
<div class="c"><div class="ch">Expert Routing</div><div class="cb" id="ebs"><p class="em">Run training first</p></div></div>
<div class="c"><div class="ch">Graph Adjacency</div><div class="cb" id="adj"><p class="em">After first round</p></div></div>
</div><div>
<div class="c"><div class="ch">Leaderboard</div><div class="cb" id="lbd"><p class="em">No ratings</p></div></div>
<div class="c"><div class="ch">Loss History</div><div class="cb"><canvas id="lc" height="170" style="width:100%"></canvas></div></div>
<div class="c"><div class="ch">Event Log</div><div class="cb" style="padding:0"><div class="la" id="elog"></div></div></div>
</div></div></div>
<div class="pan" id="t-play"><div class="g2"><div>
<div class="c"><div class="ch">Board 6x6</div>
<div class="cb" style="display:flex;justify-content:center;padding:.8rem"><svg id="bs" viewBox="0 0 320 320" style="max-width:380px;width:100%"></svg></div>
<div class="cb" style="border-top:1px solid var(--brd)"><table class="mt">
<tr><td>To Play</td><td id="pt">-</td></tr><tr><td>Move</td><td id="pm">0</td></tr><tr><td>Passes</td><td id="pp">0</td></tr><tr><td>Result</td><td id="pr">-</td></tr></table>
<div style="margin-top:.6rem;display:flex;gap:.4rem;flex-wrap:wrap">
<button class="btn s" onclick="pNew()">New</button><button class="btn s o" onclick="pPass()">Pass</button><button class="btn s o" onclick="pAI()">AI Move</button><button class="btn s o" onclick="pAuto()">Autoplay</button></div></div></div>
</div><div>
<div class="c"><div class="ch">Expert Analysis</div><div class="cb" id="pex"><p class="em">Start a game</p></div></div>
<div class="c"><div class="ch">Moves</div><div class="cb" id="pmv"><p class="em">No moves</p></div></div></div></div></div>
<div class="pan" id="t-work"><div class="g2"><div>
<div class="c"><div class="ch">Workers</div><div class="cb" id="wl"><p class="em">No workers</p></div>
<div class="cb" style="border-top:1px solid var(--brd)">
<label class="fld">Add worker<div style="display:flex;gap:.35rem"><input type="text" id="nwu" placeholder="http://192.168.1.50:8765" style="flex:1"/><button class="btn s" onclick="addW()">Add</button></div></label>
<div style="display:flex;gap:.4rem"><button class="btn s o" onclick="pingW()">Ping All</button><button class="btn s o" onclick="scanW()">Scan LAN</button></div></div></div>
<div class="c"><div class="ch">Setup</div><div class="cb">
<div class="step"><span class="step-n">1</span><span class="step-t">On each worker: <code style="color:var(--gold)">python microgo.py worker</code></span></div>
<div class="step"><span class="step-n">2</span><span class="step-t">Add the worker URL above, or Scan LAN.</span></div>
<div class="step"><span class="step-n">3</span><span class="step-t">Go to Dashboard and click Start.</span></div></div></div>
</div><div>
<div class="c"><div class="ch">Cluster</div><div class="cb"><table class="mt">
<tr><td>Device</td><td id="wd">-</td></tr><tr><td>Local IPs</td><td id="wi">-</td></tr><tr><td>Local Worker</td><td id="wlc">-</td></tr><tr><td>Token</td><td id="wt">-</td></tr></table></div></div></div></div></div>
<div class="pan" id="t-conf"><div class="g2"><div>
<div class="c"><div class="ch">Training</div><div class="cb">
<label class="fld">Rounds<input type="number" id="cr" value="6" min="1"/></label>
<label class="fld">Self-play sims<input type="number" id="cs" value="20" min="1"/></label>
<label class="fld">Games/job<input type="number" id="cg" value="2" min="1"/></label>
<label class="fld">Jobs/round/net<input type="number" id="cj" value="2" min="1"/></label>
<label class="fld">Train steps<input type="number" id="ct" value="60" min="1"/></label>
<label class="fld">Batch size<input type="number" id="ccb" value="64" min="8"/></label>
<label class="fld">LR<input type="text" id="cl" value="0.001"/></label>
<label class="fld">Eval sims<input type="number" id="ce" value="28" min="1"/></label>
<label class="fld">Eval games<input type="number" id="ceg" value="8" min="2"/></label>
<button class="btn" onclick="saveCfg()">Save</button></div></div>
</div><div>
<div class="c"><div class="ch">Cluster</div><div class="cb">
<label class="fld">Token<input type="text" id="ctk"/></label>
<label class="fld">Local Worker<select id="clo"><option value="1" selected>Yes</option><option value="0">No</option></select></label>
<button class="btn" onclick="saveClus()">Save</button></div></div>
<div class="c"><div class="ch">Actions</div><div class="cb" style="display:flex;flex-direction:column;gap:.4rem">
<button class="btn o" onclick="resetN()">Reset Networks</button><button class="btn o" onclick="expSt()">Export</button></div></div></div></div></div>
"""
    # JS uses backticks for HTML-building to avoid quote hell
    js=r"""
const $=id=>document.getElementById(id);
const api=async(p,b)=>{const o=b!==undefined?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)}:{};return(await fetch(p,o)).json()};
document.querySelector('.tabs').addEventListener('click',e=>{if(e.target.tagName!=='BUTTON')return;const id=e.target.dataset.tab;document.querySelectorAll('.pan').forEach(p=>p.classList.remove('on'));document.querySelectorAll('.tabs button').forEach(b=>b.classList.remove('on'));document.getElementById('t-'+id).classList.add('on');e.target.classList.add('on');if(id==='play')pNew()});
function eBars(ew,tgt){if(!ew||!ew.length)return;const cols=['var(--gold)','var(--sky)','var(--teal)','var(--rose)'];const ns=['opening','tactics','territory','endgame'];let h='';for(let i=0;i<ew.length;i++){const p=(ew[i]*100).toFixed(1);h+=`<div class="eb"><span class="eb-l">${ns[i]}</span><div class="eb-t"><div class="eb-f" style="width:${p}%;background:${cols[i]}"></div></div><span class="eb-v">${p}%</span></div>`}$(tgt).innerHTML=h}
async function poll(){try{const d=await api('/api/status');
$('mr').textContent=d.current_round+' / '+d.cfg.rounds;$('mra').textContent=d.replay_a;$('mrb').textContent=d.replay_b;
$('mla').textContent=d.last_loss_a||'-';$('mlb').textContent=d.last_loss_b||'-';$('mar').textContent=d.a_vs_random||'-';$('mbr').textContent=d.b_vs_random||'-';$('mhh').textContent=d.h2h||'-';
if(d.is_training){$('sd').className='dot run';$('sl').textContent='training';$('bs1').disabled=true;$('bs2').disabled=false}
else{$('sd').className='dot ok';$('sl').textContent='idle';$('bs1').disabled=false;$('bs2').disabled=true}
if(d.leaderboard&&d.leaderboard.length){let h='';for(let i=0;i<d.leaderboard.length;i++){const e=d.leaderboard[i];h+=`<div class="lr"><span class="lr-k">${i+1}</span><span class="lr-n">${e[0]}</span><span class="lr-e">${Math.round(e[1])}</span></div>`}$('lbd').innerHTML=h}
eBars(d.expert_weights,'ebs');
if(d.graph&&d.graph.edges){const ns=d.graph.experts,n=ns.length;let h=`<div class="ag" style="grid-template-columns:58px repeat(${n},46px)"><div></div>`;for(const nm of ns)h+=`<div class="ah">${nm.slice(0,4)}</div>`;for(let ri=0;ri<n;ri++){h+=`<div class="ah" style="text-align:right;padding-right:3px">${ns[ri].slice(0,4)}</div>`;for(const v of d.graph.edges[ri]){const a=Math.min(1,v*2);h+=`<div class="ac" style="background:rgba(201,162,39,${a.toFixed(2)})">${v.toFixed(2)}</div>`}}h+='</div>';$('adj').innerHTML=h}
if(d.log&&d.log.length){const el=$('elog');let h='';for(const e of d.log)h+=`<div><span class="ts">${e.ts}</span>${e.msg}</div>`;el.innerHTML=h;el.scrollTop=el.scrollHeight}
if(d.round_history&&d.round_history.length){const cv=$('lc'),ctx=cv.getContext('2d');const dpr=devicePixelRatio||1;cv.width=cv.offsetWidth*dpr;cv.height=170*dpr;ctx.scale(dpr,dpr);const w=cv.offsetWidth,h=170;ctx.clearRect(0,0,w,h);const la=d.round_history.map(r=>r.loss_a).filter(v=>v!=null),lb=d.round_history.map(r=>r.loss_b).filter(v=>v!=null);if(la.length){const all=[...la,...lb],mn=Math.min(...all),mx=Math.max(...all),rng=mx-mn||1;const pad={t:8,r:8,b:20,l:34},pw=w-pad.l-pad.r,ph=h-pad.t-pad.b;ctx.strokeStyle='#262c32';ctx.lineWidth=1;for(let i=0;i<=4;i++){const y=pad.t+ph*i/4;ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(w-pad.r,y);ctx.stroke()}function line(data,col){ctx.strokeStyle=col;ctx.lineWidth=2;ctx.beginPath();data.forEach((v,i)=>{const x=pad.l+(i/Math.max(1,data.length-1))*pw,y=pad.t+(1-(v-mn)/rng)*ph;i===0?ctx.moveTo(x,y):ctx.lineTo(x,y)});ctx.stroke()}line(la,'#c9a227');line(lb,'#5b8fd4')}}
}catch(e){}}
async function startT(){await api('/api/train/start',{})}
async function stopT(){await api('/api/train/stop',{})}
async function resetN(){if(confirm('Reset?'))await api('/api/reset',{})}
async function expSt(){const d=await api('/api/export');const b=new Blob([JSON.stringify(d,null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='microgo.json';a.click()}
async function saveCfg(){await api('/api/config',{rounds:+$('cr').value,selfplay_sims:+$('cs').value,games_per_job:+$('cg').value,selfplay_jobs_per_round:+$('cj').value,train_steps:+$('ct').value,batch_size:+$('ccb').value,learning_rate:+$('cl').value,eval_sims:+$('ce').value,eval_games:+$('ceg').value})}
async function saveClus(){await api('/api/cluster',{token:$('ctk').value,include_local:$('clo').value==='1'})}
async function addW(){const u=$('nwu').value.trim();if(u){await api('/api/workers/add',{url:u});$('nwu').value='';rfW()}}
async function pingW(){await api('/api/workers/ping',{});rfW()}
async function scanW(){$('wl').innerHTML='<p class="em">Scanning...</p>';await api('/api/workers/scan',{});rfW()}
async function rfW(){const d=await api('/api/workers');$('wd').textContent=d.device;$('wi').textContent=(d.local_ips||[]).join(', ')||'-';$('wlc').textContent=d.include_local?'Yes':'No';$('wt').textContent=d.token?d.token.slice(-4):'-';
const el=$('wl');if(!d.workers||!d.workers.length){el.innerHTML='<p class="em">No workers - add one above or Scan LAN</p>';return}
let h='';for(const w of d.workers){h+=`<div class="wr"><span class="dot ${w.ok?'ok':''}"></span><span class="u">${w.url}</span><span class="ws ${w.ok?'ok':'er'}">${w.ok?'online':(w.error||'offline')}</span><button class="btn s d" data-rm="${w.url}">x</button></div>`}el.innerHTML=h}
document.addEventListener('click',e=>{if(e.target.dataset.rm){api('/api/workers/remove',{url:e.target.dataset.rm}).then(()=>rfW())}});
async function pNew(){const d=await api('/api/playground/new',{});rB(d)}
async function pPass(){const d=await api('/api/playground/move',{move:'pass'});rB(d)}
async function pAI(){const d=await api('/api/playground/ai',{});rB(d)}
async function pAuto(){for(let i=0;i<60;i++){const d=await api('/api/playground/ai',{});rB(d);if(d.board.game_over)break;await new Promise(r=>setTimeout(r,180))}}
function rB(data){if(!data||!data.board)return;const b=data.board,svg=$('bs'),S=b.size,mg=28,cs=(320-2*mg)/(S-1);
let h=`<rect x="4" y="4" width="312" height="312" rx="4" fill="#b0862a"/><rect x="7" y="7" width="306" height="306" rx="3" fill="#c08c32"/>`;
for(let i=0;i<S;i++){const p=mg+i*cs;h+=`<line x1="${mg}" y1="${p}" x2="${320-mg}" y2="${p}" stroke="#896320" stroke-width=".8"/><line x1="${p}" y1="${mg}" x2="${p}" y2="${320-mg}" stroke="#896320" stroke-width=".8"/>`}
[[1,1],[1,4],[4,1],[4,4]].forEach(([r,c])=>{h+=`<circle cx="${mg+c*cs}" cy="${mg+r*cs}" r="2.3" fill="#896320"/>`});
for(const c of b.cells){if(c.v===0)continue;const cx=mg+c.c*cs,cy=mg+c.r*cs,rd=cs*.41;if(c.v===1){h+=`<circle cx="${cx}" cy="${cy}" r="${rd}" fill="#181818" stroke="#2a2a2a" stroke-width=".5"/><circle cx="${cx-rd*.22}" cy="${cy-rd*.22}" r="${rd*.16}" fill="#333" opacity=".45"/>`}else{h+=`<circle cx="${cx}" cy="${cy}" r="${rd}" fill="#e6e2d8" stroke="#bbb" stroke-width=".5"/><circle cx="${cx-rd*.22}" cy="${cy-rd*.22}" r="${rd*.16}" fill="#fff" opacity=".55"/>`}}
if(!b.game_over)for(const lm of b.legal_moves){if(lm.r<0)continue;const cx=mg+lm.c*cs,cy=mg+lm.r*cs;const hit=b.cells.find(c=>c.r===lm.r&&c.c===lm.c);if(hit&&hit.v!==0)continue;h+=`<circle cx="${cx}" cy="${cy}" r="${cs*.28}" fill="transparent" stroke="${b.to_play===1?'#444':'#ccc'}" stroke-width="1" stroke-dasharray="3,3" opacity="0" cursor="pointer" onmouseover="this.setAttribute('opacity','.45')" onmouseout="this.setAttribute('opacity','0')" data-mv="${lm.move}"/>`}
for(let i=0;i<S;i++){const x=mg+i*cs;h+=`<text x="${x}" y="15" text-anchor="middle" fill="#896320" font-size="10" font-family="IBM Plex Mono">${String.fromCharCode(65+i)}</text><text x="11" y="${mg+i*cs+4}" text-anchor="middle" fill="#896320" font-size="10" font-family="IBM Plex Mono">${i+1}</text>`}
svg.innerHTML=h;$('pt').textContent=b.to_play_label;$('pm').textContent=b.move_count;$('pp').textContent=b.passes;
$('pr').textContent=b.game_over?(b.score_black>0?'Black +'+b.score_black.toFixed(1):'White +'+(-b.score_black).toFixed(1)):'-';
eBars(data.experts,'pex');
if(data.moves&&data.moves.length){let mh='<div class="ml">';for(let i=0;i<data.moves.length;i++)mh+=`<span class="mc ${i%2===0?'b':'w'}">${i+1}.${data.moves[i]}</span>`;mh+='</div>';$('pmv').innerHTML=mh}else $('pmv').innerHTML='<p class="em">No moves</p>'}
document.getElementById('bs').addEventListener('click',e=>{if(e.target.dataset.mv)api('/api/playground/move',{move:+e.target.dataset.mv}).then(d=>rB(d))});
poll();setInterval(poll,3000);rfW();
"""
    return build_page("EXOSFEAR MicroGo KG",badge,badge.upper(),tabs,body,js)

# === ROUTES ===
@app.route("/")
def index():return Response(worker_page() if APP_MODE=="worker" else coord_page(),mimetype="text/html")

@app.route("/health")
def health():return jsonify({"ok":True,"role":"microgo_worker","worker_name":socket.gethostname(),"device":ST.device,"version":VERSION})

@app.route("/selfplay",methods=["POST"])
def selfplay_ep():
    token=app.config.get("AUTH_TOKEN","")
    if token and request.headers.get("X-Token","")!=token:return jsonify({"ok":False,"error":"bad token"}),403
    data=request.json or {};WORKER_STATE.busy=True;WORKER_STATE._log("Job: "+str(data.get("games",0))+" games")
    t0=time.time()
    try:
        if MOCK_MODE:
            games=int(data.get("games",2));time.sleep(0.4*games)
            res={"ok":True,"results":[{"winner":random.choice([1,-1])}]*games,"samples":compress_obj([]),"avg_gate":[0.25]*4,"worker_name":socket.gethostname(),"seconds":round(time.time()-t0,2)}
            WORKER_STATE.jobs_done+=1;WORKER_STATE.total_games+=games;WORKER_STATE._log("Done: "+str(games)+"g "+str(res["seconds"])+"s (mock)")
        else:
            res=do_selfplay_job(data["model"],ST.device,int(data.get("sims",20)),int(data.get("games",2)))
            res["ok"]=True;res["worker_name"]=socket.gethostname();res["seconds"]=round(time.time()-t0,2)
            ng=len(res.get("results",[]));WORKER_STATE.jobs_done+=1;WORKER_STATE.total_games+=ng
            WORKER_STATE._log("Done: "+str(ng)+"g "+str(res["seconds"])+"s")
        WORKER_STATE.last_job_time=now_ts();return jsonify(res)
    except Exception as e:WORKER_STATE._log("FAIL: "+str(e));return jsonify({"ok":False,"error":str(e)}),500
    finally:WORKER_STATE.busy=False

@app.route("/api/worker/status")
def wk_status():
    ips=get_local_ips();port=app.config.get("WORKER_PORT",DEFAULT_WORKER_PORT)
    url=("http://"+ips[0]+":"+str(port)) if ips else ("http://localhost:"+str(port))
    return jsonify({"url":url,"device":ST.device,"token":app.config.get("AUTH_TOKEN",""),"busy":WORKER_STATE.busy,"jobs_done":WORKER_STATE.jobs_done,"total_games":WORKER_STATE.total_games,"total_samples":WORKER_STATE.total_samples,"last_job_time":WORKER_STATE.last_job_time,"started_at":WORKER_STATE.started_at,"log":list(WORKER_STATE.log)})

@app.route("/api/status")
def api_status():
    last=ST.round_history[-1] if ST.round_history else{}
    return jsonify({"is_training":ST.is_training,"current_round":ST.current_round,"cfg":ST.cfg,
        "replay_a":ST.replays["A"].size() if"A"in ST.replays else 0,"replay_b":ST.replays["B"].size() if"B"in ST.replays else 0,
        "last_loss_a":last.get("loss_a"),"last_loss_b":last.get("loss_b"),"a_vs_random":last.get("a_vs_random"),"b_vs_random":last.get("b_vs_random"),"h2h":last.get("h2h"),
        "expert_weights":last.get("expert_weights"),"expert_names":list(EXPERT_NAMES),"graph":last.get("graph"),
        "leaderboard":ST.leaderboard.top()[:10],"round_history":ST.round_history,"log":list(ST.log)})

@app.route("/api/config",methods=["POST"])
def api_config():
    data=request.json or{}
    for k in["rounds","selfplay_sims","games_per_job","selfplay_jobs_per_round","train_steps","batch_size","eval_sims","eval_games"]:
        if k in data:ST.cfg[k]=int(data[k])
    if"learning_rate"in data:ST.cfg["learning_rate"]=float(data["learning_rate"])
    ST._log("Config updated");return jsonify({"ok":True})

@app.route("/api/cluster",methods=["POST"])
def api_cluster():
    data=request.json or{}
    if"token"in data:ST.token=data["token"]
    if"include_local"in data:ST.include_local=bool(data["include_local"])
    return jsonify({"ok":True})

@app.route("/api/workers")
def api_workers():return jsonify({"workers":ping_workers(ST.worker_urls,ST.token) if ST.worker_urls else[],"device":ST.device,"local_ips":get_local_ips(),"include_local":ST.include_local,"token":ST.token})
@app.route("/api/workers/add",methods=["POST"])
def wk_add():
    url=(request.json or{}).get("url","").strip().rstrip("/")
    if url and url not in ST.worker_urls:ST.worker_urls.append(url);ST._log("Worker added: "+url)
    return jsonify({"ok":True})
@app.route("/api/workers/remove",methods=["POST"])
def wk_rm():url=(request.json or{}).get("url","").strip().rstrip("/");ST.worker_urls=[w for w in ST.worker_urls if w!=url];return jsonify({"ok":True})
@app.route("/api/workers/ping",methods=["POST"])
def wk_ping():return jsonify({"ok":True,"results":ping_workers(ST.worker_urls,ST.token)})
@app.route("/api/workers/scan",methods=["POST"])
def wk_scan():
    found=[];
    for ip in get_local_ips():found.extend(scan_subnet(ip,DEFAULT_WORKER_PORT,ST.token))
    new_urls=[];
    for f in found:
        u=f.get("url","").rstrip("/")
        if u and u not in ST.worker_urls:ST.worker_urls.append(u);new_urls.append(u)
    ST._log("Scan: "+str(len(found))+" found, "+str(len(new_urls))+" new");return jsonify({"ok":True,"found":len(found),"new":new_urls})
@app.route("/api/reset",methods=["POST"])
def api_reset():ST.init_nets();ST.round_history.clear();ST.current_round=0;return jsonify({"ok":True})
@app.route("/api/export")
def api_export():
    out={"leaderboard":ST.leaderboard.top(),"round_history":ST.round_history,"cfg":ST.cfg}
    if not MOCK_MODE:
        for n in["A","B"]:
            if n in ST.nets:out["model_"+n]=net_to_b64(ST.nets[n])
    return jsonify(out)
@app.route("/api/train/start",methods=["POST"])
def train_start():
    if ST.is_training:return jsonify({"ok":False,"error":"running"})
    if not ST.nets:ST.init_nets()
    ST.is_training=True;ST._stop_flag=False;threading.Thread(target=_train_loop,daemon=True).start();return jsonify({"ok":True})
@app.route("/api/train/stop",methods=["POST"])
def train_stop():ST._stop_flag=True;ST._log("Stop requested");return jsonify({"ok":True})
@app.route("/api/playground/new",methods=["POST"])
def pg_new():ST.playground_state=GoState.new();ST.playground_moves=[];return jsonify({"board":ST.get_board_json(ST.playground_state),"moves":[],"experts":_gexp(ST.playground_state)})
@app.route("/api/playground/move",methods=["POST"])
def pg_move():
    if not ST.playground_state:ST.playground_state=GoState.new();ST.playground_moves=[]
    mv=request.json.get("move");mv_int=PASS_MOVE if mv=="pass"else int(mv)
    ns=ST.playground_state.try_play(mv_int)
    if ns is None:return jsonify({"error":"illegal","board":ST.get_board_json(ST.playground_state),"moves":ST.playground_moves})
    ST.playground_state=ns;ST.playground_moves.append(move_to_str(mv_int))
    return jsonify({"board":ST.get_board_json(ST.playground_state),"moves":ST.playground_moves,"experts":_gexp(ST.playground_state)})
@app.route("/api/playground/ai",methods=["POST"])
def pg_ai():
    if not ST.playground_state:ST.playground_state=GoState.new();ST.playground_moves=[]
    if ST.playground_state.game_over():return jsonify({"board":ST.get_board_json(ST.playground_state),"moves":ST.playground_moves,"experts":None})
    if MOCK_MODE:mv=random.choice(ST.playground_state.legal_moves())
    else:
        net=ST.nets.get("A")
        if not net:ST.init_nets();net=ST.nets["A"]
        mv=eval_move(net,ST.device,ST.playground_state,sims=ST.cfg.get("selfplay_sims",20))
    ns=ST.playground_state.try_play(mv)
    if ns is None:ns=ST.playground_state.try_play(PASS_MOVE);mv=PASS_MOVE
    if ns:ST.playground_state=ns;ST.playground_moves.append(move_to_str(mv))
    return jsonify({"board":ST.get_board_json(ST.playground_state),"moves":ST.playground_moves,"experts":_gexp(ST.playground_state)})

def _gexp(state):
    if MOCK_MODE:
        phase=state.move_count/MAX_GAME_LEN;w=[max(.05,.4-phase*.35),max(.05,.2+phase*.15),max(.05,.2+phase*.1),max(.05,.2+phase*.2)];s=sum(w);return[x/s for x in w]
    net=ST.nets.get("A")
    if not net:return None
    try:_,_,aux=infer_aux(net,ST.device,state);return aux["weights"]
    except:return None

# === TRAINING LOOP ===
def _train_loop():
    try:_mock_train() if MOCK_MODE else _real_train()
    except Exception as e:ST._log("Error: "+str(e))
    finally:ST.is_training=False

def _mock_train():
    for ri in range(1,ST.cfg["rounds"]+1):
        if ST._stop_flag:ST._log("Stopped");break
        ST.current_round=ri;ST._log("=== Round "+str(ri)+"/"+str(ST.cfg["rounds"])+" (demo) ===")
        for n in["A","B"]:
            ST._log("  Self-play "+n);results=push_selfplay_parallel("mock",ST.cfg["selfplay_sims"],ST.cfg["selfplay_jobs_per_round"]*ST.cfg["games_per_job"],ST.worker_urls,ST.token,ST.include_local,ST.device)
            for res in results:ST._log("    "+res.get("worker_name","?")+" "+str(len(res.get("results",[])))+"g "+str(res.get("seconds",0))+"s");ST.replays[n].add([None]*len(res.get("results",[])))
        la=max(.3,4.5-ri*.4+random.uniform(-.2,.2));lb=max(.3,4.6-ri*.38+random.uniform(-.2,.2));ST._log("  Train A:"+str(round(la,4))+" B:"+str(round(lb,4)))
        phase=ri/ST.cfg["rounds"];ew=[max(.05,.4-phase*.35),max(.05,.2+phase*.15),max(.05,.2+phase*.1),max(.05,.2+phase*.2)];s=sum(ew);ew=[w/s for w in ew]
        edges=[[random.uniform(.1,.5) for _ in range(4)] for _ in range(4)]
        for row in edges:s=sum(row);row[:]=[v/s for v in row]
        wa=random.randint(2,6);avr=min(1,.3+ri*.08+random.uniform(-.05,.05));bvr=min(1,.3+ri*.07+random.uniform(-.05,.05))
        ST.leaderboard.update("A_current","B_current",wa/8)
        ST.round_history.append({"round":ri,"loss_a":round(la,4),"loss_b":round(lb,4),"a_vs_random":round(avr,3),"b_vs_random":round(bvr,3),"h2h":str(wa)+"/8","expert_weights":ew,"graph":{"experts":list(EXPERT_NAMES),"edges":edges,"temp":1.0-ri*.05}})
        ST._log("  Eval: A wins "+str(wa)+"/8");ST.leaderboard.ensure("A_r"+str(ri).zfill(3),ST.leaderboard.ratings.get("A_current",1000));ST.leaderboard.ensure("B_r"+str(ri).zfill(3),ST.leaderboard.ratings.get("B_current",1000))
    ST._log("Done")

def _real_train():
    cfg=ST.cfg;total=cfg["selfplay_jobs_per_round"]*cfg["games_per_job"]
    for ri in range(1,cfg["rounds"]+1):
        if ST._stop_flag:ST._log("Stopped");break
        ST.current_round=ri;ST._log("=== Round "+str(ri)+"/"+str(cfg["rounds"])+" ===")
        for nn in["A","B"]:
            ST._log("Self-play "+nn+" ("+str(total)+"g)");mb=net_to_b64(ST.nets[nn])
            for res in push_selfplay_parallel(mb,cfg["selfplay_sims"],total,ST.worker_urls,ST.token,ST.include_local,ST.device):
                if res.get("samples"):
                    packed=decompress_obj(res["samples"]);samps=[Sample(np.array(i[0],dtype=np.float32),np.array(i[1],dtype=np.float32),float(i[2])) for i in packed];ST.replays[nn].add(samps)
                ST._log("  "+nn+"."+res.get("worker_name","?")+" "+str(len(res.get("results",[])))+"g "+str(res.get("seconds",0))+"s")
        ST._log("Training...");info={"round":ri}
        for n in["A","B"]:
            tr=train_team(ST.nets[n],ST.replays[n],ST.device,cfg["train_steps"],cfg["batch_size"],cfg["learning_rate"])
            info["loss_"+n.lower()]=tr.get("total_loss")
            if tr["total_loss"]:ST._log("  "+n+":"+str(round(tr["total_loss"],4)))
            if tr.get("graph"):info["graph"]=tr["graph"]
        ST._log("Eval...");ev=evaluate_pair(ST.nets,ST.device,cfg["eval_sims"],cfg["eval_games"])
        sa=ev["wins_A"]/max(1,ev["games"]);ST.leaderboard.update("A_current","B_current",sa)
        info["a_vs_random"]=ev.get("A_vs_random");info["b_vs_random"]=ev.get("B_vs_random");info["h2h"]=str(ev["wins_A"])+"/"+str(ev["games"])
        try:_,_,aux=infer_aux(ST.nets["A"],ST.device,GoState.new());info["expert_weights"]=aux["weights"]
        except:pass
        ST.round_history.append(info);ST._log("  A wins "+str(ev["wins_A"])+"/"+str(ev["games"]))
        ST.leaderboard.ensure("A_r"+str(ri).zfill(3),ST.leaderboard.ratings.get("A_current",1000));ST.leaderboard.ensure("B_r"+str(ri).zfill(3),ST.leaderboard.ratings.get("B_current",1000))
    ST._log("Done")

# === CLI ===
def banner(lines):
    w=max(len(l) for l in lines)+4;print("\n"+"="*(w+4))
    for l in lines:print("  "+l)
    print("="*(w+4))

def run_coordinator(port=DEFAULT_COORD_PORT,token="",device=None,interactive=True):
    global APP_MODE;APP_MODE="coordinator"
    if device:ST.device=device
    ST.token=token;ST.init_nets()
    if interactive:port=resolve_port(port,"Coordinator")
    ips=get_local_ips()
    ip_lines=["            http://"+ip+":"+str(port) for ip in ips]
    banner(["EXOSFEAR MicroGo KG -- COORDINATOR","","  Dashboard:  http://localhost:"+str(port)]+ip_lines+["  Device:     "+ST.device,"  Mode:       "+("DEMO" if MOCK_MODE else "LIVE"),"  Token:      "+(token or"none"),"","  1. Open dashboard in browser","  2. Workers tab -> add workers or Scan LAN","  3. Dashboard -> Start"])
    app.config["AUTH_TOKEN"]=token;app.config["WORKER_PORT"]=port
    app.run(host="0.0.0.0",port=port,debug=False,use_reloader=False)

def run_worker(port=DEFAULT_WORKER_PORT,token="",device=None,interactive=True):
    global APP_MODE;APP_MODE="worker"
    if device:ST.device=device
    if interactive:port=resolve_port(port,"Worker")
    ips=get_local_ips();urls=["http://"+ip+":"+str(port) for ip in ips]
    banner(["EXOSFEAR MicroGo KG -- WORKER","","  Share this with coordinator:"]+["  -> "+u for u in urls]+["","  Device: "+ST.device,"  Token:  "+(token or"none"),"","  On coordinator machine:","    python microgo.py coordinator","    Add this URL in Workers tab","    Start training"])
    WORKER_STATE._log("Worker started on port "+str(port))
    app.config["AUTH_TOKEN"]=token;app.config["WORKER_PORT"]=port
    app.run(host="0.0.0.0",port=port,debug=False,use_reloader=False)

def interactive_menu():
    print("\n  EXOSFEAR MicroGo KG Distributed\n")
    print("  1) Coordinator  (main machine)")
    print("  2) Worker       (helper machine)")
    print("  Mode: "+("DEMO (no torch)" if MOCK_MODE else "LIVE (torch found)")+"\n")
    try:choice=input("  Select [1]: ").strip() or"1"
    except:choice="1"
    if choice in("1","c"):
        token=secrets.token_urlsafe(8)
        try:
            raw=input("  Token [Enter="+token+"]: ").strip()
            if raw:token=raw
        except:pass
        run_coordinator(token=token)
    elif choice in("2","w"):
        token=""
        try:token=input("  Token (match coordinator) [none]: ").strip()
        except:pass
        run_worker(token=token)

def main():
    p=argparse.ArgumentParser(description="EXOSFEAR MicroGo KG")
    p.add_argument("mode",nargs="?",default="menu",choices=["menu","coordinator","worker"])
    p.add_argument("--port",type=int,default=None);p.add_argument("--token",default="");p.add_argument("--device",default=None)
    a=p.parse_args()
    if a.mode=="coordinator":run_coordinator(port=a.port or DEFAULT_COORD_PORT,token=a.token,device=a.device,interactive=a.port is None)
    elif a.mode=="worker":run_worker(port=a.port or DEFAULT_WORKER_PORT,token=a.token,device=a.device,interactive=a.port is None)
    else:interactive_menu()

if __name__=="__main__":main()
