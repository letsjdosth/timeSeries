import cmath

import numpy as np

def difference_oper(sequence: list):
    before = sequence[0]
    iterator_for_seq = iter(sequence)
    next(iterator_for_seq)
    diff_sequence = []
    for now in iterator_for_seq:
        diff_sequence.append(now - before)
        before = now
    return diff_sequence

def ar_polynomial_roots(phi_samples: list[list[float]]) -> list[list[tuple[float, float]]]:
    # return [[(r, \theta), (r, \theta),...]] with increasing order w.r.t r
    def sort_key1(c):
        return c[0]
    
    ar_poly_polar_roots_at_samples = []
    for sample in phi_samples:
        coeff = [1] + [-x for x in sample]
        ar_poly = np.polynomial.polynomial.Polynomial(coeff)
        ar_poly_roots = ar_poly.roots()
        ar_poly_polar_roots = [cmath.polar(x) for x in ar_poly_roots] # r,\theta
        ar_poly_polar_roots.sort(key=sort_key1, reverse=False)
        ar_poly_polar_roots_at_samples.append(ar_poly_polar_roots)
    print("# of AR roots: ", len(ar_poly_polar_roots_at_samples[0]))
    return ar_poly_polar_roots_at_samples

if __name__=="__main__":
    test_seq = [1,2,3,4,5,7]
    print(difference_oper(test_seq))