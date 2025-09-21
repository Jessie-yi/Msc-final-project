# styles/colors.py —— Single Source of Truth for colors

# Discrete tokens (qualitative)
TOKENS = {
    "adm": {  # admission type
        "Emergency": "#D55E00",   # Vermillion
        "Waiting":   "#F0E442",   # Mustard Yellow
        "Planned":   "#009E73",   # Teal Green
        "Other":     "#56B4E9",   # Sky Blue
    },
    "gender": {
        "Male":   "#0074B2",      # (new) Male blue
        "Female": "#DB4898",      # (new) Female magenta
        "Unknown":"#999999",
    },
    "totals": {
        "FAE_total": "#000000",   # Black
        "FCE_total": "#FF9DA7",   # Soft Pink
    },
    "timeliness": {
        "MeanWait":   "#8C564B",  # Muted Brown
        "MedianWait": "#7F7F7F",  # Graphite Grey
        "MeanLOS":    "#BCBD22",  # Olive
        "MedianLOS":  "#17BECF",  # Cyan
    },
}

SERIES_COLOR = {
    "Emergency share": TOKENS["adm"]["Emergency"],
    "Waiting share":   TOKENS["adm"]["Waiting"],
    "Planned share":   TOKENS["adm"]["Planned"],
    "Other share":     TOKENS["adm"]["Other"],
    "Male share":      TOKENS["gender"]["Male"],
    "Female share":    TOKENS["gender"]["Female"],
    "Unknown":         TOKENS["gender"]["Unknown"],
    "FAE total":       TOKENS["totals"]["FAE_total"],
    "FCE total":       TOKENS["totals"]["FCE_total"],
    "Mean wait":       TOKENS["timeliness"]["MeanWait"],
    "Median wait":     TOKENS["timeliness"]["MedianWait"],
    "Mean LOS":        TOKENS["timeliness"]["MeanLOS"],
    "Median LOS":      TOKENS["timeliness"]["MedianLOS"],
}

# Continuous color scales (diverging / sequential) aligned with tokens
SCALES = {
    # Percent of total: Planned → White → Waiting
    "total_pct": [(0.0, TOKENS["adm"]["Planned"]), (0.5, "#FFFFFF"), (1.0, TOKENS["adm"]["Waiting"])],

    # %Male palettes
    "pct_male_fb": [(0.0, TOKENS["gender"]["Female"]), (0.5, "#FFFFFF"), (1.0, TOKENS["gender"]["Male"])],  # default
    "pct_male_rg": [(0.0, "#B22222"), (0.5, "#F7F7F7"), (1.0, "#2E8B57")],  # Red→Green (备用)
    "pct_male_to": [(0.0, TOKENS["adm"]["Planned"]), (0.5, "#FFFFFF"), (1.0, TOKENS["adm"]["Waiting"])],    # Teal→Yellow (备用)
}
