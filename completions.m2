-- Given tensor dimensions, this program finds for each number of observed entries
-- the number of partial tensors which are finitely completable
 
dims = [2, 2, 4]

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

print("preparatory steps ready")

-- Compute the rank of the Jacobian corresponding to all entries
Jrank = rank J

print("rank of full Jacobian computed")

ncompletable = new MutableList from (for i from 1 to nentries list 0)
ntensors = new MutableList from (for i from 1 to nentries list 0)
for i from 1 to nentries do (
    S = subsets(tensorentries, i);
    ntensors#(i - 1) = ntensors#(i - 1) + (length S);
    ncompletable#(i - 1) = ncompletable#(i - 1) + number(S, s -> rank(jacobian(ideal s)) == Jrank);
    << ncompletable#(i - 1);
    << "/";
    << ntensors#(i - 1);
    << " of ";
    << replace(///\[|\]///, "", replace(", ", "x", toString(dims)));
    << (if ndims == 1 then "x1" else "");
    << " tensors with ";
    << i;
    << " observed entries are finitely completable\n";
    << flush;
)

