from sage.all import *
from itertools import combinations
from collections import defaultdict

# Define the polynomial variable 'x' in the global scope
R = PolynomialRing(QQ, 'x')
x = R.gen()

### ---- TOURNAMENT TOOLS ----

def seidel_matrix(T):
    A = T.adjacency_matrix()
    return A - A.transpose()

def are_partial_transpose_equivalent_by_scc_union(T1, T2, verbose=False):
    """
    Checks if T1 and T2 are equivalent under any sequence of partial transposes
    on unions of strongly connected components (SCCs), where each step transposes
    the block corresponding to the running union of SCCs in T1.
    Returns True if such a sequence exists, False otherwise.
    """
    if T1.order() != T2.order():
        return False # Tournaments of different orders cannot be equivalent

    # Optimized check: if the characteristic polynomials are different, they can't be isomorphic.
    # No need to pass 'x' here for degree or characteristic_polynomial, Sage handles it.
    if seidel_matrix(T1).characteristic_polynomial() != seidel_matrix(T2).characteristic_polynomial():
        return False

    adj = T1.adjacency_matrix()
    sccs = T1.strongly_connected_components()
    n = T1.order()
    
    visited_indices = set() 
    
    sorted_sccs = sorted(sccs, key=lambda s: min(s))

    for scc in sorted_sccs:
        visited_indices.update(scc) 
        
        new_mat = Matrix(adj)
        
        for i_orig in visited_indices:
            for j_orig in visited_indices:
                new_mat[i_orig, j_orig] = adj[j_orig, i_orig]
        
        if verbose:
            print(f"Transposing on indices {list(sorted(list(visited_indices)))}")
            print(DiGraph(new_mat).adjacency_matrix())

        if DiGraph(new_mat).is_isomorphic(T2):
            if verbose:
                print("Found isomorphism after partial transpose.")
            return True
    
    if T1.is_isomorphic(T2):
        return True

    if verbose:
        print("No isomorphism found under any SCC-union partial transpose.")
    return False

### ---- FILTER TOOLS ----

def irreducible_components_signature(T):
    comps = T.strongly_connected_components_subgraphs()
    return sorted([str(G.adjacency_matrix()) for G in comps])

def could_be_partial_transpose(T1, T2):
    return irreducible_components_signature(T1) == irreducible_components_signature(T2)

def is_strongly_connected(T):
    return T.is_strongly_connected()

# Build McKay matrix (2n x 2n) from the Seidel matrix of tournament S
def mckay_matrix(S): 
    S_seidel = seidel_matrix(S)
    nrows, ncols = S_seidel.nrows(), S_seidel.ncols()
    D_S = zero_matrix(2 * nrows, 2 * ncols)
    for i in range(nrows):
        for j in range(ncols):
            if S_seidel[i, j] == 1:
                D_S[2 * i, 2 * j] = 1
                D_S[2 * i, 2 * j + 1] = 0
                D_S[2 * i + 1, 2 * j] = 0
                D_S[2 * i + 1, 2 * j + 1] = 1
            elif S_seidel[i, j] == -1:
                D_S[2 * i, 2 * j] = 0
                D_S[2 * i, 2 * j + 1] = 1
                D_S[2 * i + 1, 2 * j] = 1
                D_S[2 * i + 1, 2 * j + 1] = 0
            else:
                D_S[2 * i, 2 * j] = 0
                D_S[2 * i, 2 * j + 1] = 0
                D_S[2 * i + 1, 2 * j] = 0
                D_S[2 * i + 1, 2 * j + 1] = 0
    return D_S

# Check if two McKay matrices (converted to digraphs) are isomorphic
def mckay_check(T1, T2): 
    if T1.order() != T2.order():
        return False
    
    # No need to pass 'x' here for characteristic_polynomial
    if seidel_matrix(T1).characteristic_polynomial() != seidel_matrix(T2).characteristic_polynomial():
        return False

    mckay_T1_matrix = mckay_matrix(T1)
    mckay_T2_matrix = mckay_matrix(T2)
    
    D_S_A = DiGraph(mckay_T1_matrix)
    D_S_B = DiGraph(mckay_T2_matrix)
    
    return D_S_A.is_isomorphic(D_S_B)


### ---- MAIN ENTRY ----

def test_partial_transpose_for_bad_polys(bad_polys):
    """
    Input: list of Sage polynomials where per-switching fails.
    Returns: A dictionary where keys are polynomials and values are lists of
             tournaments that cannot be realized by partial transposes or switching
             equivalence from any other tournament in their class.
    Also prints, for each class, the proportion of tournaments that can be generated
    through switching and partial transpose, and for each order, the same proportion.
    """
    grouped_tournaments = defaultdict(list)
    all_checked_orders = set(p.degree() for p in bad_polys)
    singular_tournaments = defaultdict(list)
    order_stats = defaultdict(lambda: {'total': 0, 'singular': 0})

    for n in sorted(list(all_checked_orders)):
        print(f"\n🔢 Generating tournaments of order {n}...")
        for T in digraphs.tournaments_nauty(n):
            poly = seidel_matrix(T).characteristic_polynomial()
            grouped_tournaments[poly].append(T)

    for i, poly in enumerate(bad_polys, 1):
        n = poly.degree()
        class_tourns = grouped_tournaments.get(poly, [])
        total_in_class = len(class_tourns)
        order_stats[n]['total'] += total_in_class

        if total_in_class < 2:
            print(f"\n⏭️ Skipping charpoly #{i}: {poly} (order {n}) - not enough tournaments in this class.")
            continue

        print(f"\n🔍 Checking charpoly #{i}/{len(bad_polys)} (order {n})")
        print(poly)

        temp_remaining_tourns = list(class_tourns)
        remaining_after_switching = []

        while temp_remaining_tourns:
            current_T = temp_remaining_tourns.pop(0)
            remaining_after_switching.append(current_T)
            new_temp_remaining = []
            for other_T in temp_remaining_tourns:
                if not mckay_check(current_T, other_T):
                    new_temp_remaining.append(other_T)
            temp_remaining_tourns = new_temp_remaining

        print(f"  After switching equivalence, {len(remaining_after_switching)} tournament(s) remain.")

        final_remaining_tournaments = []
        temp_remaining_pt = list(remaining_after_switching)

        while temp_remaining_pt:
            current_T = temp_remaining_pt.pop(0)
            final_remaining_tournaments.append(current_T)
            new_temp_remaining_pt = []
            for other_T in temp_remaining_pt:
                if not are_partial_transpose_equivalent_by_scc_union(current_T, other_T):
                    new_temp_remaining_pt.append(other_T)
            temp_remaining_pt = new_temp_remaining_pt

        num_singular = len(final_remaining_tournaments)
        order_stats[n]['singular'] += num_singular

        print(f"  After partial transpose, {num_singular} tournament(s) remain.")

        if num_singular:
            print(f"⚠️ Found {num_singular} tournament(s) in this class that cannot be realized by switching or partial transpose:")
            for T in final_remaining_tournaments:
                print(f"  - Singular Tournament Adjacency Matrix:\n{T.adjacency_matrix()}")
            singular_tournaments[poly] = final_remaining_tournaments
        else:
            print("🎉 No singular tournaments in this class.")

        # Proportion reporting for this class
        prop = 1 - (num_singular / total_in_class if total_in_class else 0)
        print(f"  Proportion of tournaments in this class that can be generated: {100*prop:.3f}% (charpoly: {poly})")

    # print("\n=== Polynomials and their singular tournaments ===")
    if not singular_tournaments:
        print("All polynomials' tournament classes are fully realizable by switching or partial transpose.")
    # else:
    #     for poly, tournaments in singular_tournaments.items():
    #         print(f"\nPolynomial: {poly}")
    #         for T in tournaments:
    #             print(f"  - Singular Tournament Adjacency Matrix:\n{T.adjacency_matrix()}")

    # print("\n=== Proportion of singular tournaments by order (within bad poly classes) ===")
    # for n in sorted(order_stats):
    #     total = order_stats[n]['total']
    #     singular = order_stats[n]['singular']
    #     prop = 1 - (singular / total if total else 0)
    #     print(f"Order {n}: {100*float(prop):.3f}% of tournaments in bad poly classes can be generated (1 - {singular}/{total})")

    print("\n=== Overall proportion of singular tournaments by order (all tournaments) ===")
    for n in sorted(order_stats):
        # Count all non-isomorphic tournaments of order n
        all_tourns = list(digraphs.tournaments_nauty(n))
        total = len(all_tourns)
        singular = order_stats[n]['singular']
        prop = 1 - (singular / total if total else 0)
        print(f"Order {n}: {100*float(prop):.3f}% of tournaments can be generated (1 - {singular}/{total})")


    return singular_tournaments

