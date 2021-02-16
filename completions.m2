-- Given dimensions and number of observed entries for a partial tensor,
-- this program finds the number of partial tensors which are finitely
-- completable from the observed entries
 
dims = [2, 2, 2]

ndims = length dims
-- Make sure dimensions are valid
assert(ndims > 0)
for d in dims do assert(d > 0)
assert(ndims == 1 or number(dims, d -> d == 1) == 0)

nentries = fold((a, b) -> a*b, dims)

-- Create a polynomial ring of the parameters
R = QQ[x_1..x_nentries]

-- Create the parameter vectors whose outer product equals the complete tensor
k = 1
params = for i from 0 to (ndims - 1) list (
    for j from 1 to dims#i list x_k do k = k + 1
)

-- get all sequences of parameters whose product corresponds to an entry
-- if ndims >= 2, take the cartesian product of ndims sets, otherwise put each element to its own set
entryparams = (if length(params) == 1
    then for param in params#0 list { param }
    else fold((a, b) -> for c in (a**b) list flatten c, params)
)

-- get the entry parametrisations by taking products of parameters
tensorentries = for seq in entryparams list (fold((a,b) -> a * b, seq)) 

I = ideal tensorentries
J = jacobian I

-- Compute the rank of the Jacobian corresponding to all entries
Jrank = rank J

-- For all numbers of observed entries, form all possible sets of observed entries
S = for nobserved from 1 to nentries list subsets(tensorentries, nobserved)
ntensors = for s in S list length s

-- Get the ranks of all Jacobians corresponding to partial tensors
ranks = for s in S list (for tsr in s list rank(jacobian(ideal tsr)))
ncompletable = for nobserved from 1 to nentries list number(ranks#(nobserved - 1), r -> r == Jrank)

for i from 1 to nentries do (
    << ncompletable#(i - 1) << "/" << ntensors#(i - 1) << " of " << toString(dims) << " tensors with " << i << " observed entries are finitely completable\n"
)


