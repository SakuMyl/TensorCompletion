-- Given tensor dimensions, this program finds for each number of observed entries
-- the number of partial tensors which are finitely completable
 
dims = {3, 3, 3}

ndims = #dims
-- Make sure dimensions are valid
assert(ndims > 0)
for d in dims do assert(d > 0)
assert(ndims == 1 or number(dims, d -> d == 1) == 0)

nentries = product(dims)
nvars = sum(dims)

-- Create the parameter vectors whose outer product equals the complete tensor
k = 0
paramIndices = for i from 0 to (ndims - 1) list (
    for j from 1 to dims#i list k do k = k + 1
)

-- get all sequences of parameters whose product corresponds to an entry
-- if ndims >= 2, take the cartesian product of ndims sets, otherwise put each element to its own set
entryparamIndices = (if #paramIndices == 1
    then for param in paramIndices#0 list { param }
    else fold((a, b) -> for c in (a**b) list flatten c, paramIndices)
)

x = for i from 1 to nentries list random(QQ);

tensorentries = for seq in entryparamIndices list product(apply(seq, k -> x#(k)))

J = mutableMatrix(QQ, nvars, nentries)
for j from 0 to nentries - 1 do (
    params = entryparamIndices#j;
    for i in params do (
        J_(i, j) = tensorentries#j / x#i;
    );
);

getJacobianRank = (entryIndices) -> (
    J2 = submatrix(J, toList(entryIndices));
    return rank(J2);
)

isFinitelyCompletable = (entryIndices, fullJacobianRank) -> (
    r = getJacobianRank(entryIndices);
    return r == fullJacobianRank;
)

-- Compute the rank of the Jacobian corresponding to all entries
Jrank = getJacobianRank(toList(0..(nentries - 1)))

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
    completable = isFinitelyCompletable(locations, Jrank);
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
    << replace(///\{|\}///, "", replace(", ", "x", toString(dims)));
    << (if ndims == 1 then "x1" else "");
    << " tensors with ";
    << i;
    << " observed entries are finitely completable\n";
    << flush;
)
