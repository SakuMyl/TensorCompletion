-- Given tensor dimensions, this program finds for each number of observed entries
-- the number of partial tensors which are finitely completable
 
dims = {2, 2, 3}
dims = sort(dims)
dimprods = accumulate((a, b) -> a * b, 1, drop(dims, 1));
ndims = #dims
-- Make sure dimensions are valid
assert(ndims > 0)
for d in dims do assert(d > 0)
assert(ndims == 1 or number(dims, d -> d == 1) == 0)

nentries = product(dims)
nvars = sum(dims)

-- A contains the maximal pairs of indices (i, j) for which dims#i = ... = dims#j
A = {}
firstIndex = 0
prev = 0
for i from 0 to #dims do (
    d = if i == #dims then 0 else dims#i;
    if prev != d then (
        if i - firstIndex > 1 then (
            A = append(A, (firstIndex, i - 1));
        );
        firstIndex = i;
    );
    prev = d;
)
taupermIndices = new MutableList from (for i from 1 to #dims list 0)
for i from 0 to #dims - 1 do (
    j = -1;
    for k from 0 to #A - 1 do (
        pair = A#k;
        if pair#0 <= i and pair#1 >= i then (
            j = k;
            break;
        );
    );
    taupermIndices#i = j;
)
taupermIndices = toList(taupermIndices);

-- Create the parameter vectors whose outer product equals the complete tensor
k = 0
paramIndices = for i from 0 to (ndims - 1) list (
    for j from 1 to dims#i list k do k = k + 1
)

cartProductRec = (input, output, current, depth) -> (
    if depth == #input then (
        output = append(output, current);
        return output;
    );
    n = #(input#depth);
    for i from 0 to n - 1 do (
        current = append(current, (input#depth)#i);
        output = cartProductRec(input, output, current, depth + 1);
        current = drop(current, -1);
    );
    return output;
)

cartProduct = (input) -> (
    current = {};
    output = {};
    return cartProductRec(input, output, current, 0);
)

-- get all sequences of parameters whose product corresponds to an entry
-- if ndims >= 2, take the cartesian product of ndims sets, otherwise put each element to its own set
-- entryparamIndices = cartProduct(paramIndices, true)
entryparamIndices = cartProduct(paramIndices)

x = for i from 1 to nentries list random(QQ);

tensorentries = for seq in entryparamIndices list product(apply(seq, k -> x#(k)))

J = mutableMatrix(QQ, nvars, nentries)
for j from 0 to nentries - 1 do (
    params = entryparamIndices#j;
    for i in params do (
        J_(i, j) = tensorentries#j / x#i;
    );
);

indexToEntryParamIndex = (k) -> (
    multiplier = 1;
    ret = 0;
    for i from 0 to #k - 1 do (
        j = #k - 1 - i;
        ret = ret + multiplier * k#j;
        multiplier = multiplier * dims#j;
    );
    return ret;
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

nOfTensors = 1 + sum(for i from 1 to nentries list binomial(nentries, i))
progress = 0
ndots = 0


D = cartProduct(for d in dims list toList(0..(d - 1)))
sigmaperms = for d in dims list permutations(d);
tauperms = for pair in A list permutations(toList(pair#0..pair#1));
perms = cartProduct(join(sigmaperms, tauperms));
Tperms = new MutableList from (for i from 1 to #perms list {});
for i from 1 to nentries do (
    ntensors = binomial(nentries, i);
    S = set(subsets(D, i));
    ncompletable = 0;
    while #S > 0 do (
        T = (elements S)#0;
        for p from 0 to #perms - 1 do (
            permset = perms#p;
            -- permset contains for each index of entries of T a permutation
            -- which consists of sigma and tau permutations
            ret = new MutableList from (for i from 1 to #T list {});
            for j from 0 to #T - 1 do (
                entry = T#j;
                permutedEntry = new MutableList from (for i from 1 to #dims list 0);
                for k from 0 to #dims - 1 do (
                    taupermIndex = taupermIndices#k;
                    sigma = permset#k;
                    if taupermIndex == -1 then (
                        permutedEntry#k = sigma#(entry#k);
                    ) else (
                        l = (A#taupermIndex)#0;
                        tau = permset#(taupermIndex + #dims);
                        permutedEntry#k = sigma#(entry#(tau#(k - l)));
                    );
                );
                ret#j = toList(permutedEntry);
            );
            Tperms#p = toList(ret);
        );
        temp = #S;
        S = S - set(toList(Tperms));
        nUniqueTensors = temp - #S;
        entryIndices = apply(T, k -> indexToEntryParamIndex(k));
        if isFinitelyCompletable(entryIndices, Jrank) then ncompletable = ncompletable + nUniqueTensors;
    );
    << ncompletable;
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
