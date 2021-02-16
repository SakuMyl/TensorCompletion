-- Given dimensions and number of observed entries for a partial tensor,
-- this program finds the number of partial tensors which are finitely
-- completable from the observed entries
 
dims = [2, 2, 2]

ndims = length dims
-- Make sure dimensions are valid
assert(ndims > 0)
for d in dims do assert(d > 0)

nentries = fold((a, b) -> a*b, dims)
-- Make sure the given number of observed entries is valid
assert(nobserved >= 0)
assert(nobserved <= nentries)

-- Create the parameter vectors
k = 1
params = for i from 0 to (ndims - 1) list (
    for j from 1 to dims#i list x_k do k = k + 1
)
-- Create a polynomial ring of the parameters
R = QQ(params)

-- get all sequences of parameters whose product corresponds to an entry
entryparams = fold((a, b) -> for c in (a**b) list flatten c, params)

-- get the entries by taking products
tensorentries = for seq in entryparams list (fold((a,b) -> (a_R)*(b_R), seq)) 
I = ideal tensorentries

J = jacobian I
-- Compute the rank of the Jacobian with all entries
Jrank = rank J

-- For all numbers of observed entries, form all possible sets of observed entries
S = for nobserved from 0 to ndims list subsets(tensorentries, nobserved)
ntensors = for s in S length s

-- Get the ranks of all Jacobians corresponding to partial tensors
ranks = for s in S list (for tsr in s list rank(jacobian(ideal s)))
ncompletable = for number(ranks, r -> r == Jrank)

-- for s in S do (
--     I_ = ideal s
--     J_ = jacobian I_
--     r = rank J_
--     if r == Jrank then ncompletable = ncompletable + 1
-- )

<< ncompletable << " of " << ntensors << " tensors with " << nobserved << " observed entries are finitely completable\n"


