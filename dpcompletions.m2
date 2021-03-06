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

isFinitelyCompletable = (params, fullJacobianRank) -> (
    r = getPartialJacobianRank(params);
    return r == fullJacobianRank;
)

getPartialJacobianRank = params -> (
    return rank(jacobian(ideal params));
)


ncompletable = new MutableList from (for i from 1 to nentries list 0)
currentSize = 1;
locations = new MutableList from { 0 };
nOfTensors = 1 + sum(for i from 1 to nentries list binomial(nentries, i));
progress = 0;
ndots = 0;

while not currentSize == 0 do (
    newDots = floor((numeric progress) / nOfTensors * 100 - ndots);
    ndots = ndots + newDots;
    scan(1 .. newDots, i -> << "." << flush);
    n = currentSize - 1;
    locationsCopy = toList locations;
    partialEntries = tensorentries_locationsCopy;
    completable = isFinitelyCompletable(partialEntries, Jrank);
    if completable then (
        entriesLeft = nentries - (last locations) - 1;
        for i from currentSize to nentries do (
            nCompletableToAdd = binomial(entriesLeft, i - currentSize);
            ncompletable#(i - 1) = ncompletable#(i - 1) + nCompletableToAdd;
            progress = progress + nCompletableToAdd;
        );
        if locations#(n) == nentries - 1 then (
            currentSize = currentSize - 1;
            locations = drop(locations, -1);
            if n > 0 then (
                locations#(n - 1) = locations#(n - 1) + 1;
            );
        ) else (
            locations#(n) = locations#(n) + 1;
        );
    ) else (
        progress = progress + 1;
        if locations#(n) == nentries - 1 then (
            currentSize = currentSize - 1;
            locations = drop(locations, -1);
            if n > 0 then (
                locations#(n - 1) = locations#(n - 1) + 1;
            );
        ) else (
            locations = append(locations, locations#(n) + 1);
            currentSize = currentSize + 1;
        );
    );
)
<< "\n";

for i from 1 to nentries do (
    ntensors = binomial(nentries, i);
    << ncompletable#(i - 1);
    << "/";
    << ntensors;
    << " of ";
    << replace(///\[|\]///, "", replace(", ", "x", toString(dims)));
    << (if ndims == 1 then "x1" else "");
    << " tensors with ";
    << i;
    << " observed entries are finitely completable\n";
    << flush;
)
