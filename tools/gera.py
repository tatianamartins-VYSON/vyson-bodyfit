import numpy as np,json,base64,sys
T='makehuman/makehuman/data/targets/'
V=[];faces=[];g=None;J={}
for l in open('makehuman/makehuman/data/3dobjs/base.obj'):
    if l.startswith('v '):V.append([float(x) for x in l.split()[1:4]])
    elif l.startswith('g '):g=l.split()[1]
    elif l.startswith('f ') and g.startswith('joint'):J.setdefault(g,set()).update(int(x.split('/')[0])-1 for x in l.split()[1:])
    elif l.startswith('f ') and g=='body':faces.append([int(x.split('/')[0])-1 for x in l.split()[1:]])
V=np.array(V)
def tgt(p,w):
    for l in open(T+p):
        if l[0].isdigit():i,x,y,z=l.split();V2[int(i)]+=w*np.array([float(x),float(y),float(z)])
for sexo,lst in {'f':[('macrodetails/caucasian-female-young.target',1),('macrodetails/universal-female-young-averagemuscle-averageweight.target',1),('breast/female-young-averagemuscle-averageweight-maxcup-averagefirmness.target',.35)],
                 'm':[('macrodetails/caucasian-male-young.target',1),('macrodetails/universal-male-young-averagemuscle-averageweight.target',1)]}.items():
    V2=V.copy()
    for p,w in lst:tgt(p,w)
    # suaviza mamilos (manequim neutro)
    adj={}
    for f in faces:
        for a in f:
            adj.setdefault(a,set()).update(f)
    H=V2[:,1].max()-V2[:,1].min();y0_=V2[:,1].min()
    for s_ in(-1,1):
        cand=[i for i in adj if s_*V2[i,0]>.3 and s_*V2[i,0]<1.6 and .66<(V2[i,1]-y0_)/H<.78]
        top=max(cand,key=lambda i:V2[i,2])
        reg=[i for i in cand if np.linalg.norm(V2[i]-V2[top])<.45]
        for _ in range(60):
            nv={i:np.mean([V2[j] for j in adj[i] if j!=i],0) for i in reg}
            for i in reg:V2[i]=nv[i]
    # virilha neutra
    reg=[i for i in adj if abs(V2[i,0])<.9 and .40<(V2[i,1]-y0_)/H<.52 and V2[i,2]>V2[:,2].mean()]
    for _ in range(40):
        nv={i:np.mean([V2[j] for j in adj[i] if j!=i],0) for i in reg}
        for i in reg:V2[i]=nv[i]
    used=sorted({i for f in faces for i in f});mp={o:n for n,o in enumerate(used)}
    P=V2[used]*10  # dm->cm
    y0=V2[used][:,1].min()*10;P[:,1]-=y0
    L={k[6:]:[round(float(c),2) for c in (V2[list(v)].mean(0)*10-[0,y0,0])] for k,v in J.items() if not any(t in k for t in('toe','finger','foot-','tooth','tongue'))}
    tri=[]
    for f in faces:
        f=[mp[i] for i in f];tri+= [f[0],f[1],f[2]]+([f[0],f[2],f[3]] if len(f)==4 else [])
    print(sexo,len(P),len(tri)//3,'altura',P[:,1].max().round(1))
    json.dump({'pos':base64.b64encode(P.astype('<f4').tobytes()).decode(),'lm':L,'idx':base64.b64encode(np.array(tri,'<u4').tobytes()).decode()},open(f'corpo_{sexo}.json','w'))
