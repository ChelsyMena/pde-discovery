ideal_eq = {
        'u': 0.0,
        'u_x': 0.0,
        'u_xx': -1.0,
        'u_xxx': 0.0,
        'u_xxxx': -1.0,
        'u_xxxxx': 0.0,
        'u_xxxxxx': 0.0,
        'u²': 0.0,
        'u³': 0.0,
        '(u²)_x': -0.5,
        '(u²)_xx': 0.0,
        '(u³)_x': 0.0,
        'u·u_x': 0.0,
        'u·u_xx': 0.0,
    }

found_eq = {
        '1': -0.209, 
        'u': 0,
        'u_x': 4.735,
        'u_xx': -0.146,
        'u_xxx': 0.407,
        'u_xxxx': -0.883235,
        'u_xxxxx': 0.0,
        'u_xxxxxx': 0.0,
        'u²': 0.0,
        'u³': 0.0,
        '(u²)_x': -0.426486,
        '(u²)_xx': -0.114,
        '(u³)_x': 0.0,
        'u·u_x': -0.100,
        'u·u_xx': 0.0,
    }


def l1_distance(eq1: dict, eq2: dict) -> float:
    """
    Compute the L1 distance between two coefficient dictionaries.
    
    Handles missing keys (treats them as 0) and sums absolute differences.
    
    Args:
        eq1: first equation dict {term_name: coefficient}
        eq2: second equation dict {term_name: coefficient}
    
    Returns:
        L1 distance = sum(|coeff1 - coeff2|) for all terms
    """
    all_terms = set(eq1.keys()) | set(eq2.keys())
    distance = sum(abs(eq1.get(term, 0.0) - eq2.get(term, 0.0)) for term in all_terms)
    return distance


def true_positive_rate(ideal_eq: dict, found_eq: dict, tol: float = 1e-6) -> float:
    """
    Compute the true positive rate for term discovery.
    
    Formula: TP / (TP + FN + FP)
    - TP: terms present (non-zero within tol) in both equations
    - FN: terms present in ideal_eq but absent in found_eq
    - FP: terms present in found_eq but absent in ideal_eq
    
    Args:
        ideal_eq: ground truth equation dict
        found_eq: discovered equation dict
        tol: threshold for considering a coefficient "present" (default 1e-6)
    
    Returns:
        TPR in range [0, 1]; returns 0 if no terms to evaluate
    """
    # identify "present" terms (coefficient magnitude > tol) in each equation
    ideal_present = {term for term, coeff in ideal_eq.items() if abs(coeff) > tol}
    found_present = {term for term, coeff in found_eq.items() if abs(coeff) > tol}
    
    tp = len(ideal_present & found_present)  # intersection: both present
    fn = len(ideal_present - found_present)  # in ideal but not in found
    fp = len(found_present - ideal_present)  # in found but not in ideal
    
    denominator = tp + fn + fp
    if denominator == 0:
        return 0.0
    
    return (tp / denominator)*100

distance = l1_distance(ideal_eq, found_eq)
tpr = true_positive_rate(ideal_eq, found_eq, tol=1e-2)
print(f"L1 distance: {distance:.6f}")
print(f"True positive rate: {tpr:.3f}")