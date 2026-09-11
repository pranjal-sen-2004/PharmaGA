"""
PharmaGA: A Niching Multi-Objective Genetic Algorithm with Surrogate-Assisted Fitness
for Personalised Drug Selection.
"""

from __future__ import annotations
import math, random, warnings
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    _LGB_OK = True
except ImportError:
    _LGB_OK = False


@dataclass
class Drug:
    drug_id: int
    name: str
    fingerprint: np.ndarray      # 128-dim Morgan-like binary
    admet: np.ndarray            # [hepatotox, cardiotox, nephrotox, idiosync]
    targets: list
    auc_base: float = 0.5
    cyp_enzymes: list = field(default_factory=list)
    ddi_profile: np.ndarray = None


@dataclass
class PatientProfile:
    patient_id: int
    age: int
    bmi: float
    egfr: float
    cyp_phenotypes: dict         # {cyp_id: 0=PM,1=IM,2=NM,3=UM}
    hla_risk: float
    indication: str
    genomic_vector: np.ndarray = None



def generate_drug_library(n_drugs=200, n_targets=40, seed=42):
    rng = np.random.default_rng(seed)
    drugs = []
    for i in range(n_drugs):
        fp   = rng.integers(0, 2, size=128).astype(float)
        admet= rng.uniform(0.05, 0.60, size=4)
        targs= list(rng.integers(0, n_targets, size=rng.integers(1,5)))
        auc  = float(rng.beta(2,3)*0.7+0.15)
        cyp  = list(rng.integers(0,6, size=rng.integers(1,3)))
        ddi  = rng.uniform(-0.1, 0.1, size=6)
        drugs.append(Drug(i, f"Drug_{i:04d}", fp, admet, targs, auc, cyp, ddi))
    return drugs


def generate_patient(patient_id, indication='HTN', seed=None):
    rng = np.random.default_rng(seed if seed is not None else patient_id*31)
    ph  = {c: int(rng.choice([0,1,2,3], p=[0.1,0.2,0.55,0.15])) for c in range(1,7)}
    gvec= np.array([v/3.0 for v in ph.values()])
    return PatientProfile(
        patient_id=patient_id,
        age=int(rng.integers(30,80)),
        bmi=float(rng.uniform(18,42)),
        egfr=float(rng.uniform(30,120)),
        cyp_phenotypes=ph,
        hla_risk=float(rng.beta(1,8)),
        indication=indication,
        genomic_vector=gvec,
    )



def oracle_fitness(regimen, patient):
    if not regimen:
        return np.array([1.0,1.0,1.0,1.0])
    # f1 efficacy
    eff = 1.0 - np.prod([1.0-d.auc_base for d,_ in regimen])
    gmod = 0.0
    for d,_ in regimen:
        for cyp in d.cyp_enzymes:
            pt = patient.cyp_phenotypes.get(cyp%6+1, 2)
            gmod += (pt-2)*0.03
    eff = float(np.clip(eff + gmod/max(len(regimen),1), 0.0, 1.0))
    # f2 toxicity
    tox = float(np.clip(np.mean([d.admet.mean() for d,_ in regimen]) + patient.hla_risk*0.15, 0,1))
    # f3 PK mismatch
    pk = 0.0
    for d,_ in regimen:
        for cyp in d.cyp_enzymes:
            pt = patient.cyp_phenotypes.get(cyp%6+1, 2)
            if pt==0: pk+=0.3
            elif pt==1: pk+=0.1
    pk = float(np.clip(pk/max(len(regimen),1),0,1))
    # f4 DDI
    drugs=[d for d,_ in regimen]; ddi=0.0
    for i in range(len(drugs)):
        for j in range(i+1,len(drugs)):
            ddi += len(set(drugs[i].cyp_enzymes)&set(drugs[j].cyp_enzymes))*0.08
    ddi = float(np.clip(ddi,0,1))
    return np.array([-eff, tox, pk, ddi])


def encode_regimen(regimen, patient, max_k=4):
    vec=[]
    for i in range(max_k):
        if i<len(regimen):
            d,dose=regimen[i]
            vec.extend(d.fingerprint[:16].tolist())
            vec.extend(d.admet.tolist())
            vec.append(dose/4.0)
        else:
            vec.extend([0.0]*21)
    vec.extend(patient.genomic_vector.tolist())
    return np.array(vec)



class GBSurrogate:
    def __init__(self):
        self.models_lo=[]
        self.models_hi=[]
        self.models_mid=[]
        self.is_fitted=False
        self.n_obj=4
        self._sx=None; self._sy=None

    def update(self, X, Y):
        if self._sx is None:
            self._sx, self._sy = X.copy(), Y.copy()
        else:
            self._sx = np.vstack([self._sx, X])
            self._sy = np.vstack([self._sy, Y])
        if len(self._sx)>=10 and _LGB_OK:
            self._fit()

    def _fit(self):
        self.models_lo,self.models_hi,self.models_mid=[],[],[]
        p=dict(n_estimators=80,learning_rate=0.1,num_leaves=15,verbosity=-1,n_jobs=1)
        for obj in range(self.n_obj):
            y=self._sy[:,obj]
            for alpha,store in [(0.1,self.models_lo),(0.9,self.models_hi),(0.5,self.models_mid)]:
                m=lgb.LGBMRegressor(objective="quantile",alpha=alpha,**p)
                m.fit(self._sx,y)
                store.append(m)
        self.is_fitted=True

    def predict(self, x):
        if not self.is_fitted or not _LGB_OK:
            return np.full(self.n_obj,0.5), np.full(self.n_obj,1.0)
        X=x.reshape(1,-1)
        mid=np.array([m.predict(X)[0] for m in self.models_mid])
        lo =np.array([m.predict(X)[0] for m in self.models_lo])
        hi =np.array([m.predict(X)[0] for m in self.models_hi])
        return mid, np.abs(hi-lo)



def _cluster_by_targets(drugs):
    visited=set(); clusters=[]
    for d in drugs:
        if d.drug_id in visited: continue
        cluster=[d]; visited.add(d.drug_id)
        for other in drugs:
            if other.drug_id in visited: continue
            if set(d.targets)&set(other.targets):
                cluster.append(other); visited.add(other.drug_id)
        clusters.append(cluster)
    return clusters


def mcx(p1, p2, max_k=4):
    all_drugs={d.drug_id:(d,dose) for d,dose in p1+p2}
    items=list(all_drugs.values())
    if len(items)<=1: return list(p1),list(p2)
    clusters=_cluster_by_targets([d for d,_ in items])
    random.shuffle(clusters)
    dose_map={d.drug_id:dose for d,dose in items}
    c1,c2=[],[]; s1,s2=set(),set()
    for i,cluster in enumerate(clusters):
        for d in cluster:
            tup=(d,dose_map[d.drug_id])
            if i%2==0 and d.drug_id not in s1 and len(c1)<max_k:
                c1.append(tup); s1.add(d.drug_id)
            elif d.drug_id not in s2 and len(c2)<max_k:
                c2.append(tup); s2.add(d.drug_id)
    return c1 or list(p1[:1]), c2 or list(p2[:1])



def _tanimoto(a,b):
    ab=float(np.dot(a,b))
    return ab/(float(np.sum(a))+float(np.sum(b))-ab+1e-9)


def mutate(regimen, library, dose_bins=5, pm=0.15):
    if not regimen: return regimen
    reg=list(regimen); op=random.random()
    if op<0.40:
        idx=random.randrange(len(reg)); d0,dose=reg[idx]
        cands=[d for d in library if d.drug_id!=d0.drug_id and _tanimoto(d0.fingerprint,d.fingerprint)>=0.25]
        if cands: reg[idx]=(random.choice(cands),dose)
    elif op<0.70:
        idx=random.randrange(len(reg)); d,dose=reg[idx]
        reg[idx]=(d,int(np.clip(dose+random.choice([-1,1]),1,dose_bins)))
    else:
        max_k=4
        if len(reg)<max_k and random.random()<0.5:
            nd=random.choice(library)
            if nd.drug_id not in {d.drug_id for d,_ in reg}:
                reg.append((nd,random.randint(1,dose_bins)))
        elif len(reg)>1:
            reg.pop(random.randrange(len(reg)))
    return reg



def dominates(a,b):
    return bool(np.all(a<=b) and np.any(a<b))


def fast_nds(F):
    n=len(F); dc=[0]*n; ds=[[] for _ in range(n)]; fronts=[[]]
    for i in range(n):
        for j in range(n):
            if i==j: continue
            if dominates(F[i],F[j]): ds[i].append(j)
            elif dominates(F[j],F[i]): dc[i]+=1
        if dc[i]==0: fronts[0].append(i)
    k=0
    while fronts[k]:
        nf=[]
        for i in fronts[k]:
            for j in ds[i]:
                dc[j]-=1
                if dc[j]==0: nf.append(j)
        k+=1; fronts.append(nf)
    return [f for f in fronts if f]


def crowding_dist(front, F):
    n=len(front)
    if n<=2: return np.full(n,np.inf)
    dist=np.zeros(n)
    for obj in range(F.shape[1]):
        order=np.argsort([F[i,obj] for i in front])
        dist[order[0]]=np.inf; dist[order[-1]]=np.inf
        rng=F[front[order[-1]],obj]-F[front[order[0]],obj]
        if rng==0: continue
        for k in range(1,n-1):
            dist[order[k]]+=(F[front[order[k+1]],obj]-F[front[order[k-1]],obj])/rng
    return dist



class PharmaGA:
    def __init__(self, drug_library, mu=50, lam=50, max_generations=80,
                 oracle_budget=500, k_max=4, dose_bins=5, archive_cap=100,
                 tau_init=0.25, beta=0.3, pm_init=0.15,
                 surrogate_oracle_fraction=0.33, seed=42, verbose=True):
        self.library=drug_library; self.mu=mu; self.lam=lam; self.G=max_generations
        self.B=oracle_budget; self.k_max=k_max; self.dose_bins=dose_bins
        self.archive_cap=archive_cap; self.tau=tau_init; self.beta=beta
        self.pm=pm_init; self.target_frac=surrogate_oracle_fraction; self.verbose=verbose
        random.seed(seed); np.random.seed(seed)
        self.surrogate=GBSurrogate()
        self.oracle_calls=0; self.surrogate_calls=0
        self.history=[]

    def _init_pop(self, patient):
        pop=[]; n30=max(1,self.mu*3//10); n30b=max(1,self.mu*3//10)
        for _ in range(n30):
            k=random.randint(1,self.k_max); ds=random.sample(self.library,k)
            pop.append([(d,random.randint(1,self.dose_bins)) for d in ds])
        sl=sorted(self.library,key=lambda d:-d.auc_base)
        for i in range(min(n30b,len(sl))):
            pop.append([(sl[i],random.randint(1,self.dose_bins))])
        rem=self.mu-len(pop)
        for _ in range(rem):
            k=random.randint(2,self.k_max); chosen=[]; cov=set()
            cands=list(self.library); random.shuffle(cands)
            for d in cands:
                if len(chosen)>=k: break
                if set(d.targets)-cov or not chosen:
                    chosen.append(d); cov|=set(d.targets)
            pop.append([(d,random.randint(1,self.dose_bins)) for d in chosen])
        return pop[:self.mu]

    def _eval_one(self, reg, patient, force_oracle=False):
        if not self.surrogate.is_fitted or force_oracle:
            self.oracle_calls+=1
            return oracle_fitness(reg,patient), True
        x=encode_regimen(reg,patient,self.k_max)
        fhat,sigma=self.surrogate.predict(x)
        if np.max(sigma)>self.tau and self.oracle_calls<self.B:
            self.oracle_calls+=1
            return oracle_fitness(reg,patient), True
        self.surrogate_calls+=1
        return fhat, False

    def _eval_pop(self, pop, patient, force_oracle=False):
        Xs,Ys,flags=[],[],[]
        for reg in pop:
            f,used=self._eval_one(reg,patient,force_oracle)
            Xs.append(encode_regimen(reg,patient,self.k_max))
            Ys.append(f); flags.append(used)
        return np.array(Xs), np.array(Ys), flags

    def _genomic_sens(self, pop, patient):
        sens=np.zeros(len(pop))
        for i,reg in enumerate(pop):
            for d,_ in reg:
                for cyp in d.cyp_enzymes:
                    pt=patient.cyp_phenotypes.get(cyp%6+1,2)
                    if pt in (0,3): sens[i]+=1.0
        return sens

    def _tournament(self, pop, F, ranks, gs):
        sel=[]; idx=list(range(len(pop)))
        for _ in range(self.lam):
            a,b=random.sample(idx,2)
            if ranks[a]<ranks[b]: sel.append(a)
            elif ranks[b]<ranks[a]: sel.append(b)
            else: sel.append(a if gs[a]>=gs[b] else b)
        return sel

    def _env_select(self, cpop, cF, cgs):
        fronts=fast_nds(cF); np_,nf_,ng_=[],[],[]
        for front in fronts:
            if len(np_)+len(front)<=self.mu:
                for idx in front: np_.append(cpop[idx]); nf_.append(cF[idx]); ng_.append(cgs[idx])
            else:
                rem=self.mu-len(np_)
                gs_f=np.array([cgs[i] for i in front])
                cd=crowding_dist(front,cF)+0.1*(gs_f/max(gs_f.max(),1e-9))
                order=np.argsort(-cd)[:rem]
                for pos in order:
                    idx=front[pos]; np_.append(cpop[idx]); nf_.append(cF[idx]); ng_.append(cgs[idx])
                break
        return np_, np.array(nf_), np.array(ng_)

    def _update_archive(self, arch, arch_f, pop, F):
        combined=arch+pop; cF=arch_f+[F[i] for i in range(len(pop))]
        cF_arr=np.array(cF)
        fronts=fast_nds(cF_arr); nd=fronts[0] if fronts else []
        ndp=[combined[i] for i in nd]; ndf=[cF[i] for i in nd]
        if len(ndp)>self.archive_cap:
            cd=crowding_dist(list(range(len(ndp))),np.array(ndf))
            keep=np.argsort(-cd)[:self.archive_cap]
            ndp=[ndp[i] for i in keep]; ndf=[ndf[i] for i in keep]
        return ndp, ndf

    def _hv2d(self, arch_f, ref=None):
        if not arch_f: return 0.0
        if ref is None: ref=np.array([1.1,1.1])
        pts=np.array([f[:2] for f in arch_f])
        pts=pts[np.all(pts<=ref,axis=1)]
        if len(pts)==0: return 0.0
        pts=pts[np.argsort(pts[:,0])]
        hv=0.0; prev=ref[1]
        for p in pts:
            w=ref[0]-p[0]; h=prev-p[1]
            if h>0 and w>0: hv+=w*h
            prev=min(prev,p[1])
        return float(hv)

    def run(self, patient):
        pop=self._init_pop(patient)
        Xs,Ys,_=self._eval_pop(pop,patient,force_oracle=True)
        self.surrogate.update(Xs,Ys)
        fronts=fast_nds(Ys); ranks=[0]*len(pop)
        for r,front in enumerate(fronts):
            for idx in front: ranks[idx]=r
        arch,arch_f=self._update_archive([],[],pop,Ys)
        self.history=[]

        for gen in range(1,self.G+1):
            if self.oracle_calls>=self.B:
                break
            gs=self._genomic_sens(pop,patient)
            pidx=self._tournament(pop,Ys,ranks,gs)
            offspring=[]
            for i in range(0,len(pidx)-1,2):
                c1,c2=mcx(pop[pidx[i]],pop[pidx[i+1]],self.k_max)
                offspring.extend([mutate(c1,self.library,self.dose_bins,self.pm),
                                   mutate(c2,self.library,self.dose_bins,self.pm)])
            offspring=offspring[:self.lam]
            Xo,Yo,uf=self._eval_pop(offspring,patient)
            # adapt tau
            of=sum(uf)/max(len(uf),1)
            self.tau=float(np.clip(self.tau*(1.1 if of>self.target_frac+0.05 else 0.9 if of<self.target_frac-0.05 else 1.0),0.05,2.0))
            # surrogate update
            oidx=[i for i,u in enumerate(uf) if u]
            if oidx: self.surrogate.update(Xo[oidx],Yo[oidx])
            # mutation rate
            nd_o=fast_nds(Yo); r_nd=len(nd_o[0])/max(len(offspring),1) if nd_o else 0
            self.pm=float(np.clip(self.pm*math.exp(self.beta*(r_nd-0.2)),0.01,0.5))
            # env select
            cpop=pop+offspring; cF=np.vstack([Ys,Yo])
            cgs=self._genomic_sens(cpop,patient)
            pop,Ys,gs=self._env_select(cpop,cF,cgs)
            fronts=fast_nds(Ys); ranks=[0]*len(pop)
            for r,front in enumerate(fronts):
                for idx in front: ranks[idx]=r
            arch,arch_f=self._update_archive(arch,arch_f,pop,Ys)
            hv=self._hv2d(arch_f)
            self.history.append({'gen':gen,'hv':hv,'archive_size':len(arch),
                                  'oracle_calls':self.oracle_calls,
                                  'surrogate_calls':self.surrogate_calls,
                                  'pm':self.pm,'tau':self.tau})
            if self.verbose and gen%10==0:
                print(f"  Gen {gen:3d} | HV={hv:.4f} | Archive={len(arch)} | Oracle={self.oracle_calls}/{self.B} | pm={self.pm:.3f}")
        return arch, arch_f

    def personalisation_fidelity(self, archive, arch_f, patient, n_twins=10):
        if not archive: return 0.0
        non_portable=0
        for _ in range(n_twins):
            twin=generate_patient(patient.patient_id+random.randint(1,9999),
                                  patient.indication,seed=random.randint(0,9999))
            twin.age=patient.age; twin.bmi=patient.bmi; twin.egfr=patient.egfr
            tf=np.array([oracle_fitness(r,twin) for r in archive])
            twin_nd=set(fast_nds(tf)[0]) if fast_nds(tf) else set()
            non_portable+=sum(1 for i in range(len(archive)) if i not in twin_nd)
        return float(non_portable/(n_twins*max(len(archive),1)))


if __name__=="__main__":
    print("=== PharmaGA Quick Test ===")
    library=generate_drug_library(100,seed=42)
    patient=generate_patient(1,'T2DM',seed=7)
    ga=PharmaGA(library,mu=30,lam=30,max_generations=40,oracle_budget=300,verbose=True)
    arch,arch_f=ga.run(patient)
    print(f"Archive: {len(arch)} | Oracle: {ga.oracle_calls} | Surrogate: {ga.surrogate_calls}")
    print(f"HV(2D): {ga._hv2d(arch_f):.4f}")
    print(f"PF:     {ga.personalisation_fidelity(arch,arch_f,patient,n_twins=5):.4f}")
