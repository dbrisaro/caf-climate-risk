
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

C_HAZARD  = "#1B4F8A"
C_EXPO    = "#1A7A5E"
C_VULN    = "#C05B1A"
C_POLICY  = "#6B2D7A"
C_BG      = "#F7F9FC"
C_HEADER  = "#0D1B2A"
C_LIGHT   = "#FFFFFF"
C_ACCENT  = "#E8EFF7"

FONT = "Ubuntu"

PILLARS = [
    {
        "title":    "CLIMATE\nHAZARDS",
        "subtitle": "What nature brings",
        "color":    C_HAZARD,
        "items": [
            "Heavy Rainfall & Floods",
            "Droughts",
            "Heatwaves",
            "Tropical Cyclones",
        ],
        "source":  "ERA5-Land · IBTrACS",
    },
    {
        "title":    "EXPOSURE",
        "subtitle": "What is at risk",
        "color":    C_EXPO,
        "items": [
            "Population size & density",
            "Built surface & volume",
            "Critical infrastructure",
            "Urban growth trends",
        ],
        "source":  "GHS-POP · GHS-BUILT · MERIT DEM",
    },
    {
        "title":    "VULNERABILITY",
        "subtitle": "How much is lost",
        "color":    C_VULN,
        "items": [
            "Economic damages (USD)",
            "Lives lost (VSL-adjusted)",
            "Service disruption",
            "Country structural risk",
        ],
        "source":  "EM-DAT · World Bank · OSM",
    },
    {
        "title":    "POLICY\nINSIGHTS",
        "subtitle": "Where to act",
        "color":    C_POLICY,
        "items": [
            "Prioritise high-risk regions",
            "Target urban planning",
            "Strengthen DRM capacity",
            "Design climate-proof assets",
        ],
        "source":  "8 countries · 263 regions",
    },
]

fig = plt.figure(figsize=(18, 10), facecolor=C_BG)
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 18)
ax.set_ylim(0, 10)
ax.axis("off")

fig.patch.set_facecolor(C_BG)

banner = FancyBboxPatch((0, 8.6), 18, 1.4,
                        boxstyle="square,pad=0",
                        fc=C_HEADER, ec="none", zorder=1)
ax.add_patch(banner)

ax.text(9, 9.35, "CARIBBEAN CLIMATE RISK ASSESSMENT",
        ha="center", va="center", fontsize=22, fontweight="bold",
        color="white", fontfamily=FONT, zorder=2)

ax.text(9, 8.82, "Sub-national Analysis  ·  8 Countries  ·  263 Regions  ·  1995 – 2025",
        ha="center", va="center", fontsize=12, color="#A8C4E0",
        fontfamily=FONT, zorder=2)

n       = len(PILLARS)
pad_x   = 0.6
gap     = 0.55
box_w   = (18 - 2*pad_x - (n-1)*gap) / n
box_h   = 6.2
box_y0  = 1.5

arrow_x_pairs = []

for i, p in enumerate(PILLARS):
    x0 = pad_x + i * (box_w + gap)
    xc = x0 + box_w / 2
    arrow_x_pairs.append((x0, x0 + box_w))

    shadow = FancyBboxPatch((x0 + 0.07, box_y0 - 0.07), box_w, box_h,
                            boxstyle="round,pad=0.18",
                            fc="#C8D6E5", ec="none", zorder=2, alpha=0.5)
    ax.add_patch(shadow)

    box = FancyBboxPatch((x0, box_y0), box_w, box_h,
                         boxstyle="round,pad=0.18",
                         fc=C_LIGHT, ec=p["color"], linewidth=2.5, zorder=3)
    ax.add_patch(box)

    header_h = 1.35
    header = FancyBboxPatch((x0 + 0.01, box_y0 + box_h - header_h + 0.01),
                            box_w - 0.02, header_h - 0.02,
                            boxstyle="round,pad=0.15",
                            fc=p["color"], ec="none", zorder=4)
    ax.add_patch(header)

    circle = plt.Circle((x0 + 0.42, box_y0 + box_h - header_h/2 + 0.05),
                        0.28, color="white", zorder=5)
    ax.add_patch(circle)
    ax.text(x0 + 0.42, box_y0 + box_h - header_h/2 + 0.05,
            str(i + 1), ha="center", va="center",
            fontsize=13, fontweight="bold", color=p["color"],
            fontfamily=FONT, zorder=6)

    ax.text(xc + 0.15, box_y0 + box_h - header_h/2 + 0.05,
            p["title"], ha="center", va="center",
            fontsize=15, fontweight="bold", color="white",
            fontfamily=FONT, zorder=5, linespacing=1.2)

    ax.text(xc, box_y0 + box_h - header_h - 0.28,
            p["subtitle"],
            ha="center", va="top",
            fontsize=10.5, color=p["color"], style="italic",
            fontfamily=FONT, zorder=4, fontweight="bold")

    ax.plot([x0 + 0.3, x0 + box_w - 0.3],
            [box_y0 + box_h - header_h - 0.48,
             box_y0 + box_h - header_h - 0.48],
            color=p["color"], lw=1.2, alpha=0.35, zorder=4)

    item_y_start = box_y0 + box_h - header_h - 0.78
    item_gap     = 0.9
    for j, item in enumerate(p["items"]):
        iy = item_y_start - j * item_gap

        dot = plt.Circle((x0 + 0.45, iy + 0.05), 0.085,
                         color=p["color"], zorder=4)
        ax.add_patch(dot)

        ax.text(x0 + 0.72, iy + 0.05, item,
                ha="left", va="center",
                fontsize=10.5, color="#1A1A2E",
                fontfamily=FONT, zorder=4)

    chip_y = box_y0 + 0.2
    chip = FancyBboxPatch((x0 + 0.25, chip_y), box_w - 0.5, 0.5,
                          boxstyle="round,pad=0.08",
                          fc=C_ACCENT, ec=p["color"], linewidth=1, zorder=4, alpha=0.9)
    ax.add_patch(chip)
    ax.text(xc, chip_y + 0.25, p["source"],
            ha="center", va="center",
            fontsize=8.5, color=p["color"], fontfamily=FONT,
            fontweight="bold", zorder=5)

arrow_y = box_y0 + box_h / 2

for i in range(n - 1):
    x_start = pad_x + i * (box_w + gap) + box_w + 0.06
    x_end   = pad_x + (i+1) * (box_w + gap) - 0.06

    ax.annotate("",
                xy=(x_end, arrow_y), xytext=(x_start, arrow_y),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=C_HEADER,
                    lw=2.4,
                    mutation_scale=22,
                ),
                zorder=10)

eq_bar = FancyBboxPatch((pad_x, 0.12), 18 - 2*pad_x, 1.2,
                        boxstyle="round,pad=0.12",
                        fc=C_HEADER, ec="none", zorder=3, alpha=0.92)
ax.add_patch(eq_bar)

eq_text = (
    "Climate Risk  =  Hazard Intensity  ×  Exposure Level  ×  Structural Vulnerability"
)
ax.text(9, 0.72, eq_text,
        ha="center", va="center",
        fontsize=13.5, color="white", fontfamily=FONT,
        fontweight="bold", zorder=4)

out = "/home/jupyter-daniela/CAF/outputs/vulnerability/fig_framework_stakeholder.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=C_BG)
print(f"Saved → {out}")
plt.close()
