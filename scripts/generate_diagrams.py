import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUT = Path("figures")
OUT.mkdir(exist_ok=True)

def setup(title):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title(title, fontsize=16, weight="bold", pad=18)
    return fig, ax

def box(ax, x, y, w, h, title, sub="", fc="#EAF1F8", ec="#34536F"):
    p = FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.04,rounding_size=0.12",
                       linewidth=1.5,edgecolor=ec,facecolor=fc)
    ax.add_patch(p)
    ax.text(x+w/2,y+h*.65,title,ha="center",va="center",fontsize=9,weight="bold")
    if sub:
        ax.text(x+w/2,y+h*.28,sub,ha="center",va="center",fontsize=8)

def arrow(ax, a, b, color="#526779", style="-|>"):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle=style,mutation_scale=13,
                                 linewidth=1.5,color=color))

# F1: Architecture
fig, ax = setup("F1. System Architecture: Phase-Aware Scheduling of Concurrent ML Workloads")

items = [
    (0.3,5.4,2.1,0.9,"Concurrent workloads","CNN layers + GEMM groups"),
    (2.8,5.4,2.1,0.9,"Workload representation","Mapping-derived groups"),
    (5.3,5.4,2.1,0.9,"Resource characterization","Compute, local BW, DRAM BW"),
    (7.8,5.4,2.1,0.9,"Scheduler","Policy-dependent assignment"),
    (10.3,5.4,2.3,0.9,"Evaluation","Modeled makespan, remaps"),
]
for item in items:
    box(ax,*item)
for x in [2.4,4.9,7.4,9.9]:
    arrow(ax,(x,5.85),(x+.4,5.85))

ax.text(6.4,4.55,"2 × 2 Network-on-Package (NoP)",ha="center",fontsize=12,weight="bold")
chips = [
    (3.8,3.0,"C1","Compute chiplet\n1.50× compute"),
    (6.1,3.0,"C2","Compute chiplet\n1.50× compute"),
    (3.8,1.55,"C3","Memory-oriented\n1.50× memory BW"),
    (6.1,1.55,"C4","Balanced\n1.25× memory capacity"),
]
for x,y,n,s in chips:
    box(ax,x,y,1.8,.9,n,s,"#E5F2E8","#39734A")
for a,b in [
    ((5.6,3.45),(6.1,3.45)),((5.6,2.0),(6.1,2.0)),
    ((4.7,3.0),(4.7,2.45)),((7.0,3.0),(7.0,2.45))
]:
    arrow(ax,a,b,"#888888","<->")

arrow(ax,(8.8,5.4),(7.85,3.9),"#9B6B20")
ax.text(9.25,4.35,"Policy-based\nchiplet assignment",
        ha="center",va="center",fontsize=9,color="#9B6B20",
        bbox=dict(boxstyle="round,pad=.3",fc="#FFF8E9",ec="#D5A34C"))

ax.text(6.5,.65,
    "Model boundary: analytical duration and resource model. NoP topology is represented, "
    "but physical NoP contention,\nmeasured migration cost, and hardware execution are not modeled.",
    ha="center",fontsize=9,color="#8A3B32",
    bbox=dict(boxstyle="round,pad=.4",fc="#FFF3F0",ec="#C98276"))
fig.tight_layout()
fig.savefig(OUT/"fig1_system_architecture.png",dpi=300,bbox_inches="tight")
fig.savefig(OUT/"fig1_system_architecture.pdf",bbox_inches="tight")
plt.close(fig)

# F3: Methodology
fig, ax = setup("F3. Experimental Methodology and Evaluation Workflow")

steps = [
    (0.35,5.15,"1. Workload inputs","ResNet-50 layers\nSynthetic / GEMM groups"),
    (3.0,5.15,"2. Mapping and profiling","Timeloop mapping\nLayer and memory profiles"),
    (5.65,5.15,"3. Group construction","Mapping-derived logical\ngroups"),
    (8.3,5.15,"4. Characterization","Compute work, local BW,\nDRAM BW, reference cycles"),
    (8.3,2.85,"5. Phase classification","Compute / memory / mixed\nfrom normalized components"),
    (5.65,2.85,"6. Scheduler evaluation","Static, RR, reactive,\nphase-aware, oracle, locality"),
    (3.0,2.85,"7. Event-driven simulation","Chiplet assignment,\ncompletion and remap events"),
    (0.35,2.85,"8. Metrics and analysis","Modeled makespan,\ncompletion, remaps, placement"),
]
for x,y,title,sub in steps:
    box(ax,x,y,2.2,1.0,title,sub)

for x in [2.55,5.2,7.85]:
    arrow(ax,(x,5.65),(x+.45,5.65))
arrow(ax,(9.4,5.15),(9.4,3.85))
for x in [8.3,5.65,3.0]:
    arrow(ax,(x,3.35),(x-.45,3.35))

ax.text(6.5,1.65,
    "Controlled comparisons: policies are compared under matched workload, "
    "profile source, and calibration conditions.",
    ha="center",fontsize=10,weight="bold",color="#34536F")
ax.text(6.5,.75,
    "Reported times are modeled makespan estimates, not physical accelerator measurements. "
    "Execution groups are logical\nmapping-derived units, not direct measurements of temporal hardware phases.",
    ha="center",fontsize=9,color="#8A3B32",
    bbox=dict(boxstyle="round,pad=.4",fc="#FFF3F0",ec="#C98276"))
fig.tight_layout()
fig.savefig(OUT/"fig3_methodology_workflow.png",dpi=300,bbox_inches="tight")
fig.savefig(OUT/"fig3_methodology_workflow.pdf",bbox_inches="tight")
plt.close(fig)

print("Generated F1 and F3 in PNG and PDF formats.")
for p in sorted(OUT.glob("fig[13]_*")):
    print(p, p.stat().st_size, "bytes")
