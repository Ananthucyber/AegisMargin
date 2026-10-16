def generate_manager_remediation(record_row, leak_probability):
    """
    Generates actionable business decisions based on physical data & risk scores.
    """
    actions = []
    
    # 1. Structural Cost Leak Check
    if record_row["unit_cost"] > record_row["list_price"]:
        actions.append({
            "severity": "CRITICAL",
            "issue": f"Base cost (\\${record_row['unit_cost']:.2f}) exceeds list price (\\${record_row['list_price']:.2f}).",
            "action": "Pause product listing immediately. Renegotiate COGS with supplier or update master list price."
        })
    
    # 2. Discount Margin Erosion
    elif record_row["discount_pct"] > 0.15 and leak_probability > 0.60:
        actions.append({
            "severity": "HIGH",
            "issue": f"Aggressive discount ({record_row['discount_pct']*100:.0f}%) driving negative gross profit.",
            "action": f"Cap max discount allowance to 10% on Product ID {record_row['product_id']}."
        })
    
    # 3. High Volume Margin Bleed
    elif record_row["quantity"] >= 4 and leak_probability > 0.50:
        actions.append({
            "severity": "MEDIUM",
            "issue": f"High volume order ({record_row['quantity']} units) compounding unit-level negative margin.",
            "action": "Restrict bulk quantity orders for low-margin SKUs until price re-indexing."
        })
        
    if not actions:
        actions.append({
            "severity": "LOW",
            "issue": "Standard transaction profile.",
            "action": "Maintain current operational pricing strategy."
        })
        
    return actions