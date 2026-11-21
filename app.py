import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="ACO Step-by-Step", page_icon="🐜", layout="wide")
CITIES = ["A", "B", "C", "D"]
DIST = np.array([[0,4,2,5],[4,0,3,2],[2,3,0,4],[5,2,4,0]], dtype=float)
POS = {"A":(0,1), "B":(2,1), "C":(0,0), "D":(2,0)}
PSEUDO = [("identify","Identifier les villes accessibles non visitées"),("probability","Calculer les probabilités de transition"),("select","Choisir la prochaine ville"),("move","Ajouter la ville au parcours"),("evaluate","Évaluer les tournées"),("evaporate","Faire évaporer les phéromones"),("deposit","Déposer les phéromones"),("done","Passer à l’itération suivante")]

def route_text(route): return " → ".join(CITIES[i] for i in route)
def length(route): return sum(DIST[a,b] for a,b in zip(route[:-1],route[1:]))
def reset(nants, niters, alpha, beta, rho, q, seed):
    s=st.session_state; s.tau=np.ones((4,4)); np.fill_diagonal(s.tau,0)
    s.ants=[dict(id=k+1,start=k%4,route=[k%4],candidates=[],probs=[],selected=None) for k in range(nants)]
    s.ant_idx=0; s.stage="identify"; s.phase="construct"; s.iteration=1; s.routes=[]; s.best_route=None; s.best_len=float("inf"); s.history=[]
    s.alpha=alpha; s.beta=beta; s.rho=rho; s.q=q; s.nants=nants; s.niters=niters; s.seed=seed; s.rng=np.random.default_rng(seed); s.message="Matrices initialisées : τ=1 et η=1/d."
def ant_now():
    s=st.session_state
    return s.ants[s.ant_idx] if s.phase=="construct" and s.ants else None
def step():
    s=st.session_state; a=ant_now()
    if s.phase=="construct" and a:
        visited=a["route"]
        if s.stage=="identify":
            a["candidates"]=[a["start"]] if len(visited)==4 else [i for i in range(4) if i not in visited]
            a["probs"]=[]; s.stage="probability"; s.message=f"Fourmi {a['id']} : candidats identifiés."
        elif s.stage=="probability":
            cur=visited[-1]; cand=a["candidates"]
            if len(visited)==4: probs=np.array([1.0])
            else:
                w=np.array([(s.tau[cur,j]**s.alpha)*(1/DIST[cur,j]**s.beta) for j in cand]); probs=w/w.sum()
            a["probs"]=probs.tolist(); s.stage="select"; s.message="Probabilités calculées et normalisées."
        elif s.stage=="select":
            pos=int(s.rng.choice(len(a["candidates"]),p=np.array(a["probs"]))); a["selected"]=a["candidates"][pos]; s.stage="move"; s.message=f"La fourmi {a['id']} choisit {CITIES[a['selected']]}."
        elif s.stage=="move":
            a["route"].append(a["selected"]); s.message="Déplacement : "+route_text(a["route"])
            if len(a["route"])==5:
                L=length(a["route"]); s.routes.append(dict(ant=a["id"],route=a["route"].copy(),length=L)); s.ant_idx+=1
                if s.ant_idx>=len(s.ants): s.phase="evaluate"; s.stage="evaluate"; s.message="Toutes les fourmis ont terminé."
                else: s.stage="identify"
            else: s.stage="identify'
    elif s.phase=="evaluate":
        best=min(s.routes,key=lambda x:x["length"])
        if best["length"]<s.best_len: s.best_len=best["length"]; s.best_route=best["route"].copy()
        s.phase="evaporate"; s.stage="evaporate"; s.message=f"Meilleure tournée de cette itération : {best['length']:.2f}."
    elif s.phase=="evaporate":
        s.tau*=1-s.rho; s.phase="deposit"; s.stage="deposit"; s.message=f"Évaporation : τ ← (1−ρ)τ, ρ={s.rho:.2f}."
    elif s.phase=="deposit":
        for item in s.routes:
            amount=s.q/item["length"]
            for i,j in zip(item["route"][:-1],item["route"][1:]): s.tau[i,j]+=amount; s.tau[j,i]+=amount
        s.history.append(s.best_len); s.phase="done"; s.stage="done"; s.message="Dépôt terminé : matrice des phéromones mise à jour."
    elif s.phase=="done":
        if s.iteration>=s.niters: s.phase="finished"; s.message="Simulation terminée."
        else:
            s.iteration+=1; s.ants=[dict(id=k+1,start=k%4,route=[k%4],candidates=[],probs=[],selected=None) for k in range(s.nants)]
            s.ant_idx=0; s.routes=[]; s.phase="construct"; s.stage="identify"; s.message="Nouvelle itération : phéromones actualisées."

def graph():
    s=st.session_state; fig=go.Figure()
    for i in range(4):
        for j in range(i+1,4):
            x0,y0=POS[CITIES[i]]; x1,y1=POS[CITIES[j]]; tau=s.tau[i,j]
            fig.add_trace(go.Scatter(x=[x0,x1],y=[y0,y1],mode="lines",line=dict(width=1.5+min(tau,8)*.55,color="#8A9AAF"),hovertemplate=f"{CITIES[i]}–{CITIES[j]}<br>d={DIST[i,j]}<br>τ={tau:.2f}<extra></extra>",showlegend=False))
            fig.add_annotation(x=(x0+x1)/2,y=(y0+y1)/2,text=f"d={DIST[i,j]:.0f}<br>τ={tau:.2f}",showarrow=False,bgcolor="rgba(255,255,255,.8)",font=dict(size=10))
    colors=["#2563EB","#DB7C26","#16A36A","#A855F7","#D9468D"]
    for k,item in enumerate(s.routes):
        r=item["route"]; fig.add_trace(go.Scatter(x=[POS[CITIES[i]][0] for i in r],y=[POS[CITIES[i]][1] for i in r],mode="lines",line=dict(color=colors[k%len(colors)],width=2,dash="dot"),name=f"Fourmi {item['ant']}"))
    a=ant_now()
    if a:
        r=a["route"]; fig.add_trace(go.Scatter(x=[POS[CITIES[i]][0] for i in r],y=[POS[CITIES[i]][1] for i in r],mode="lines+markers",line=dict(color="#E11D48",width=5),name=f"Fourmi active {a['id']}"))
    fig.add_trace(go.Scatter(x=[POS[c][0] for c in CITIES],y=[POS[c][1] for c in CITIES],mode="markers+text",text=CITIES,textposition="middle center",marker=dict(size=38,color="#DCEBFF",line=dict(color="#2563EB",width=2)),textfont=dict(size=15,color="#12345A"),name="Villes"))
    fig.update_layout(height=360,margin=dict(l=5,r=5,t=5,b=5),xaxis=dict(visible=False,range=[-.5,2.5]),yaxis=dict(visible=False,range=[-.4,1.4],scaleanchor="x"),legend=dict(orientation="h",y=-.12))
    return fig

if "ants" not in st.session_state: reset(4,5,1,2,.5,10,7)
s=st.session_state
with st.sidebar:
    st.header("Paramètres")
    n=st.slider("Nombre de fourmis",2,8,s.nants); it=st.slider("Itérations",1,20,s.niters)
    alpha=st.slider("α — phéromones",0.0,3.0,s.alpha,.1); beta=st.slider("β — heuristique",0.0,5.0,s.beta,.1)
    rho=st.slider("ρ — évaporation",0.0,.9,s.rho,.05); q=st.number_input("Q — dépôt",.1,100.0,s.q,.5); seed=st.number_input("Graine",0,99999,s.seed)
    if st.button("Réinitialiser avec ces paramètres",use_container_width=True): reset(n,it,alpha,beta,rho,q,int(seed)); st.rerun()
    st.caption("Distances fixes. Départs des fourmis : A, B, C, D, puis cycle.")
st.title("🐜 ACO — Visualiseur pas à pas")
st.caption("Ant System classique · TSP à quatre villes")
cols=st.columns(4); cols[0].metric("Itération",f"{min(s.iteration,s.niters)} / {s.niters}"); cols[1].metric("Fourmis",s.nants); cols[2].metric("Fourmi active",ant_now()["id"] if ant_now() else "—"); cols[3].metric("Meilleure distance","—" if not np.isfinite(s.best_len) else f"{s.best_len:.2f}")
left,right=st.columns([1.15,.85],gap="large")
with left:
    st.subheader("1. Graphe et parcours"); st.plotly_chart(graph(),use_container_width=True,config={"displayModeBar":False}); st.caption("Épaisseur = phéromones. Labels = distance et τ.")
with right:
    st.subheader("2. Algorithme en cours")
    line=s.stage if s.phase=="construct" else s.phase
    for key,label in PSEUDO:
        if key==line: st.markdown(f"<div style='background:#DCEBFF;color:#12345A;padding:9px;border-radius:8px;font-weight:700;margin:4px 0'>▶ {label}</div>",unsafe_allow_html=True)
        else: st.markdown(f"<div style='padding:7px;color:#64748b;border-bottom:1px solid #eee'>{label}</div>",unsafe_allow_html=True)
    st.info(s.message)
st.divider(); st.subheader("3. Matrices")
eta=np.zeros_like(DIST); eta[DIST>0]=1/DIST[DIST>0]
m1,m2,m3=st.columns(3)
with m1: st.markdown("**Distances dᵢⱼ**"); st.dataframe(pd.DataFrame(DIST,index=CITIES,columns=CITIES),use_container_width=True)
with m2: st.markdown("**Heuristique ηᵢⱼ=1/dᵢⱼ**"); st.dataframe(pd.DataFrame(eta,index=CITIES,columns=CITIES).round(2),use_container_width=True)
with m3: st.markdown("**Phéromones τᵢⱼ**"); st.dataframe(pd.DataFrame(s.tau,index=CITIES,columns=CITIES).round(2),use_container_width=True)
st.divider(); st.subheader("4. Fourmi active")
a=ant_now()
if a:
    st.write(f"Fourmi **{a['id']}** · départ {CITIES[a['start']]} · parcours `{route_text(a['route'])}`")
    if a["candidates"]:
        cur=a["route"][-1]; rows=[]
        for pos,j in enumerate(a["candidates"]): rows.append({"Candidate":CITIES[j],"τᵢⱼ":round(s.tau[cur,j],3),"ηᵢⱼ":round(1/DIST[cur,j],3),"Probabilité":f"{a['probs'][pos]*100:.1f}%" if a["probs"] else "À calculer"})
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
else: st.caption("Construction terminée pour cette itération.")
if s.routes:
    st.markdown("**Tournées terminées**"); st.dataframe(pd.DataFrame([{"Fourmi":x["ant"],"Parcours":route_text(x["route"]),"Distance":round(x["length"],2)} for x in s.routes]),use_container_width=True,hide_index=True)
explain={"identify":"Repérer les villes non visitées (ou le retour au départ).","probability":"Calculer τᵢⱼ^α × ηᵢⱼ^β puis normaliser les poids.","select":"Tirer une ville selon la distribution calculée.","move":"Déplacer la fourmi et ajouter la ville à la tournée.","evaluate":"Comparer les longueurs des tournées et garder la meilleure.","evaporate":"Multiplier les phéromones par (1−ρ).","deposit":"Déposer Q/Lₖ sur les arêtes parcourues.","done":"L’itération est terminée.","finished":"Nombre maximal d’itérations atteint."}
st.markdown("**Que se passe-t-il ?** "+explain.get(s.stage if s.phase=="construct" else s.phase,""))
st.divider(); b1,b2,b3=st.columns([1,1,2])
with b1:
    if st.button("Réinitialiser",use_container_width=True): reset(s.nants,s.niters,s.alpha,s.beta,s.rho,s.q,s.seed); st.rerun()
with b2:
    if st.button("Étape suivante ➜",type="primary",use_container_width=True,disabled=s.phase=="finished"): step(); st.rerun()
with b3:
    if s.phase=="finished": st.success(f"Terminé. Meilleure tournée : {route_text(s.best_route)} — {s.best_len:.2f}")
    else: st.caption("Chaque clic exécute une étape logique. L’étape courante est surlignée.")
if s.history:
    st.subheader("5. Évolution de la meilleure distance"); st.line_chart(pd.DataFrame({"Itération":range(1,len(s.history)+1),"Distance":s.history}).set_index("Itération")); st.success(f"Meilleur parcours : {route_text(s.best_route)} — distance {s.best_len:.2f}")
with st.expander("Formules utilisées"):
    st.latex(r"P_{ij}^{k}=\frac{(\tau_{ij})^\alpha(\eta_{ij})^\beta}{\sum_{l\in N_i^k}(\tau_{il})^\alpha(\eta_{il})^\beta}")
    st.latex(r"\eta_{ij}=\frac{1}{d_{ij}}")
    st.latex(r"\tau_{ij}\leftarrow(1-\rho)\tau_{ij}+\sum_k\Delta\tau_{ij}^k")
    st.latex(r"\Delta\tau_{ij}^k=\frac{Q}{L_k}")
    st.caption("Ant System classique, TSP symétrique, dépôt dans les deux sens.")
