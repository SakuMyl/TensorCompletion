#include <vector>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <random>
#include <limits>
#include <armadillo>
#include <iterator>
#include <stack>
#include <list>
#include <memory>
#include <unordered_set>

using namespace std;
using namespace arma;
const double EPS = 1E-9;

using Tensor = vector<vector<arma::uword>>;
using SimplifiedTensor = vector<arma::uword>;

template <typename T>
T sum(vector<T> &vec) {
    T ret = 0;
    for (auto t : vec) ret += t;
    return ret;
}

template <typename T>
T product(vector<T> &vec) {
    T ret = 1;
    for (auto t : vec) ret *= t;
    return ret;
}

template <typename T>
void cart_product_rec(
    vector<vector<T>>& in,
    vector<vector<T>>& out,
    vector<T> current,
    int depth
    ) {
    if (depth == in.size()) {
        out.push_back(current);
        return;
    }
    int n = in[depth].size();
    for (int i = 0; i < n; i++) {
        current.push_back(in[depth][i]);
        cart_product_rec(in, out, current, depth + 1);
        current.pop_back();
    }
}

template <typename T>
void cart_product(
    vector<vector<T>>& in,
    vector<vector<T>>& out
    ) {
    vector<T> current;
    cart_product_rec(in, out, current, 0);
}

int factorial(int n) {
    assert(n >= 0);
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

int binom(int n, int k) {
    if (k > n) return 0;
    if (k == 0 || k == n) return 1;
    return binom(n - 1, k - 1) + binom(n - 1, k);
}

int get_jacobian_rank(mat &full_jacobian, vector<arma::uword> &cols) {
    arma::uvec cols_vec(&cols.front(), cols.size());
    return arma::rank(full_jacobian.cols(cols_vec));
}

int is_finitely_completable(mat &full_jacobian, vector<arma::uword> &cols, int full_rank) {
    int r = get_jacobian_rank(full_jacobian, cols);
    return r == full_rank;
}

void get_dimensions(vector<int> &dims) {
    string str;
    cout << "Give tensor dimensions separated by spaces: ";
    getline(cin, str);
    istringstream is(str);
    dims.assign(std::istream_iterator<int>(is), std::istream_iterator<int>());
}

void validate_dimensions(vector<int> &dims) {
    int ndims = dims.size();
    if (ndims == 0) {
        cout << "No dimensions given, exiting." << endl;
        exit(EXIT_FAILURE);
    }
    for (auto d : dims) {
        if(d <= 0) {
            cout << "Invalid dimensions given, exiting." << endl;
            exit(EXIT_FAILURE);
        }
    }
    if (ndims != 1 && count(dims.begin(), dims.end(), 1) > 0) {
        cout << "Redundant dimensions of length 1 given, exiting." << endl;
        exit(EXIT_FAILURE);
    }
}

void print_progress(float fprogress) {
    int percentage = int(fprogress * 100);
    int bar_width = 70;
    cout << "[";
    int pos = bar_width * fprogress;
    for (int i = 0; i < pos; i++) cout << "=";
    if (pos != bar_width) cout << ">";
    for (int i = pos + 1; i < bar_width; i++) cout << " ";
    cout << "] " << percentage << " %\r" << flush;
}

int index_vec_to_int(vector<arma::uword> &long_index, vector<int> &multipliers)
{
    int ret = 0;
    for (int i = 0; i < long_index.size(); i++) {
        ret += long_index[i] * multipliers[i];
    }
    return ret;
}

int kth_index(arma::uword in, int k, vector<int> &dims, vector<int> &multipliers)
{
    return (in / multipliers[k]) % dims[k];
}

void int_to_index_vec(arma::uword in, vector<arma::uword> &out, vector<int> &multipliers, vector<int> &dims)
{
    for (int i = 0; i < multipliers.size(); i++) {
        out[i] = kth_index(in, i, dims, multipliers);
    }
}

void shorten_indices(vector<vector<arma::uword>> &tensor, vector<arma::uword> &out, vector<int> &multipliers)
{
    for (int j = 0; j < tensor.size(); j++) {
        out[j] = index_vec_to_int(tensor[j], multipliers);
    }
}

struct vecHash
{
    size_t operator()(const SimplifiedTensor &V) const {
        int hash = 0;
        for (auto &i : V) {
            int mod = i % (8 * sizeof(int));
            hash |= 1 << mod;
            // hash ^= i + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        return hash;
    }
};

int iterate_perms(
    vector<vector<int>> &perms,
    Tensor &perm_base,
    Tensor &current_perm,
    unordered_set<SimplifiedTensor, vecHash> &permuted_tensors,
    unordered_set<SimplifiedTensor, vecHash> &representatives,
    vector<int> &multipliers,
    int n_unique,
    int index
    ) {
    return 0;
    // In case some permutation already exists in the representatives, we have hit an existing orbit
    if (n_unique < 0) return -1;
    if (index == perms.size()) {
        SimplifiedTensor key(current_perm.size());
        shorten_indices(current_perm, key, multipliers);
        sort(key.begin(), key.end());
        // If representatives contains some non-identity permutation, stop looking
        if (representatives.contains(key)) return -1;
        if (permuted_tensors.contains(key)) return n_unique;
        else {
            permuted_tensors.insert(key);
            return n_unique + 1;
        }
    }
    vector<int> &perm = perms[index];
    do {
        for (int entry = 0; entry < perm_base.size(); entry++) {
            current_perm[entry][index] = perm[perm_base[entry][index]];
        }
        n_unique = iterate_perms(perms, perm_base, current_perm, permuted_tensors, representatives, multipliers, n_unique, index + 1);
        if (n_unique < 0) return -1;
    } while (next_permutation(perm.begin(), perm.end()));
    return n_unique;
}

int stabilizer_size(vector<vector<int>> &counts, vector<int> &E_bin, vector<int> &dims, vector<int> &dim_prods)
{
    int distinct_transpositions = 1;
    int distinct_perms = 1;
    for (int dim = 0; dim < dims.size(); dim++) {
        auto &dim_counts = counts[dim];
        int prev_equal_count = 0;
        for (int j = 1; j <= dims[dim]; j++) {
            if (j == dims[dim] || dim_counts[prev_equal_count] != dim_counts[j]) {
                if (j - prev_equal_count > 1) {
                    int prev_equal_slice = prev_equal_count;
                    for (int k = prev_equal_slice + 1; k <= j; k++) {
                        if (k == j) distinct_perms *= factorial(k - prev_equal_slice);
                        else {
                            int offset = dim_prods[dim] * (k - prev_equal_slice);
                            bool equal = true;
                            for (arma::uword e = 0; e < E_bin.size(); e++) {
                                if (kth_index(e, dim, dims, dim_prods) == prev_equal_slice) {
                                    if (E_bin[e] != E_bin[e + offset]) {
                                        equal = false;
                                        break;
                                    }
                                }
                            }
                            if (!equal) {
                                if (k - prev_equal_slice > 1) distinct_perms *= factorial(k - prev_equal_slice);
                                prev_equal_slice = k;
                            }
                        }
                    }
                }
                prev_equal_count = j;
            }
        }
    }
    return distinct_perms * distinct_transpositions;
}

// Checks whether adding a given entry will change the counts so that the two rules are still satisfied
bool satisfies_rules(vector<vector<int>> &counts, vector<int> &dims, vector<arma::uword> &entry, vector<int> &E_bin, vector<int> &dim_prods)
{
    // Rule 1
    for (int k = 0; k < entry.size(); k++) {
        for (int i = 1; i < dims[k]; i++) {
            int diff = counts[k][i - 1] - counts[k][i];
            if (entry[k] == i - 1) diff++;
            else if (entry[k] == i) diff--;
            if (diff < 0) {
                return false;
            }
        }
    }
    // Rules 2 and 3
    for (int k = 1; k < entry.size(); k++) {
        if (dims[k] != dims[k - 1]) continue;
        int i = 0;
        bool equal = true;
        while (equal && i < dims[k]) {
            int diff = counts[k - 1][i] - counts[k][i]; 
            if (entry[k - 1] == i) diff++;
            if (entry[k] == i) diff--;

            if (diff < 0) {
                return false;
            }
            if (diff > 0) {
                equal = false;
            }
            i++;
        }
    }
    // Rule 4
    int entry_index = index_vec_to_int(entry, dim_prods);
    for (int k = 0; k < entry.size(); k++) {
         arma::uword i = entry[k];
         if (i == 0) continue;
         if (counts[k][i] == counts[k][i - 1]) {
            int offset = dim_prods[k];
            for (arma::uword j = 0; j < E_bin.size(); j++) {
                if (kth_index(j, k, dims, dim_prods) != i) continue;
                int diff = E_bin[j - offset] - E_bin[j];
                if (j == entry_index) diff--;
                if (diff > 0) break;
                else if (diff < 0) return false;
            }
         }
    }
    return true;
}

struct graph_node {
    list<graph_node *> neighbors;
    graph_node *parent;
    arma::uword value;
    bool visited;
};

int main() {
    vector<int> dims;
    get_dimensions(dims); 
    validate_dimensions(dims);
    sort(dims.begin(), dims.end());
    int ndims = dims.size();
    vector<int> A(ndims);
    vector<int>::iterator i = dims.begin();
    while (i < dims.end()) {
        auto p = equal_range(i, dims.end(), *i);
        for (auto j = p.first; j < p.second; j++) {
            A[j - dims.begin()] = 1;
        }
        i = p.second;
    }
    vector<int> dim_prods(ndims);
    dim_prods[ndims - 1] = 1;
    for (int i = ndims - 1; i > 0; i--) {
        dim_prods[i - 1] = dim_prods[i] * dims[i];
    }
    std::string dims_string;
    for (int i = 0; i < ndims; i++) {
        dims_string += to_string(dims[i]);
        if (i != dims.size() - 1) dims_string += 'x';
    }
    if (dims.size() == 1) dims_string += "x1";
    cout << "Computing completability of " << dims_string << " partial tensors" << endl;
    print_progress(0);
    int nentries = product(dims);
    int nvars = sum(dims);
    int k = 0;
    vector<vector<int>> param_indices(ndims);
    for (int i = 0; i < ndims; i++) {
        for (int j = 0; j < dims[i]; j++) {
            param_indices[i].push_back(k);
            k++;
        }
    }
    vector<vector<int>> entryparam_indices;
    cart_product(param_indices, entryparam_indices);

    vector<double> x(nentries);
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> r(-100, 100);
    for (int i = 0; i < nentries; i++) x[i] = r(gen);
    
    vector<double> tensorentries(nentries);
    for (int i = 0; i < entryparam_indices.size(); i++) {
        vector<int> &seq = entryparam_indices[i];
        vector<double> params(seq.size());
        for (int j = 0; j < seq.size(); j++) params[j] = x[seq[j]];
        tensorentries[i] = product(params);
    }

    mat J(nvars, nentries);
    for (int j = 0; j < nentries; j++) {
        vector<int> &params = entryparam_indices[j];
        for (int i : params) {
            J(i, j) = tensorentries[j] / x[i];
        }
    }
    vector<arma::uword> all_indices(nentries);
    iota(all_indices.begin(), all_indices.end(), 0);
    int Jrank = get_jacobian_rank(J, all_indices);
    vector<int> ncompletable(nentries + 1, 0);
    int n_tensors = 0;
    for (int i = 0; i <= nentries; i++) {
        n_tensors += binom(nentries, i);
    }
    int progress = 1;
    int percentage = 0;
    int bar_width = 70;
    
    vector<arma::uword> E(1, 0);
    vector<int> E_bin(nentries, 0);
    E_bin[0] = 1;
    int M = 1;
    list<graph_node *> l;
    graph_node G = { l, nullptr, 0, true };
    graph_node *current_node = &G;
    // counts[i][j] is how many entries have the ith index equal to j
    vector<vector<int>> counts(ndims);
    for (int i = 0; i < ndims; i++) {
        vector<int> vec(dims[i], 0);
        vec[0] = 1;
        counts[i].assign(vec.begin(), vec.end());
    }
    vector<vector<arma::uword>> long_indices(nentries);
    for (int i = 0; i < nentries; i++) {
        vector<arma::uword> index_vec(ndims);
        int_to_index_vec(i, index_vec, dim_prods, dims);
        long_indices[i] = index_vec;
    }
    int iterations = 0;
    unordered_set<SimplifiedTensor, vecHash> representatives;
    vector<vector<int>> perms(ndims);
    for (int j = 0; j < ndims; j++) {
        vector<int> perm(dims[j]);
        perms[j] = perm;
    }
    int n_perms = factorial(dims[0]);
    int prev = 0;
    for (int i = 1; i <= ndims; i++) {
        n_perms *= factorial(dims[i]);
        if (i == ndims || dims[i] != dims[prev]) {
            n_perms *= factorial(i - prev);
            prev = i;
        }
    }
    do {
        iterations++;
        if (is_finitely_completable(J, E, Jrank)) {
            int orbit_size = n_perms / stabilizer_size(counts, E_bin, dims, dim_prods);
            // cout << "E.size(): " << E.size() << endl;
            // cout << orbit_size << endl; 
            // for (auto &c : counts) {
            //     for (auto &p : c) {
            //         cout << p << " ";
            //     }
            //     cout << endl;
            // }
            ncompletable[E.size()] += orbit_size;
            // Tensor perm_base(E.size());
            // for (int j = 0; j < E.size(); j++) {
            //     perm_base[j] = long_indices[E[j]];
            // }
            // Tensor current_perm(perm_base.size());
            // for (int i = 0; i < perm_base.size(); i++) {
            //     for (int j = 0; j < perm_base[i].size(); j++) {
            //         current_perm[i].push_back(perm_base[i][j]);
            //     }
            // }
            // for (int j = 0; j < ndims; j++) {
            //     iota(perms[j].begin(), perms[j].end(), 0);
            // }
            // unordered_set<SimplifiedTensor, vecHash> permuted_tensors;
            // int n_unique = iterate_perms(perms, perm_base, current_perm, permuted_tensors, representatives, dim_prods, 0, 0);
            // if (n_unique >= 0) {
            //     ncompletable[E.size()] += n_unique;
            //     progress += n_unique;
            //     representatives.insert(E);
            // }
        } 
        vector<arma::uword> new_entries;
        arma::uword last_added = E.back();
        int first_index = last_added / dim_prods[0];
        // Loop through potential new entries to add satisfying the rules
        for (arma::uword i = last_added + 1; i < nentries && long_indices[i][0] <= first_index + 1; i++) {
            bool is_neighbor = false;
            for (graph_node *vertex : current_node->neighbors) {
                if (vertex->value == i) {
                    is_neighbor = true;
                    break;
                }
            }
            if (!is_neighbor && satisfies_rules(counts, dims, long_indices[i], E_bin, dim_prods)) {
                new_entries.push_back(i);
            }
        }
        // Add new vertices to visit
        for (int i = 0; i < new_entries.size(); i++) {
            list<graph_node *> l;
            graph_node *vertex = new graph_node();
            vertex->neighbors = l;
            vertex->parent = current_node;
            vertex->value = new_entries[i];
            vertex->visited = false;
            current_node->neighbors.push_back(vertex);
        }
        arma::uword new_entry;
        graph_node *new_vertex = nullptr;
        bool found = false;
        while (!found && current_node && !E.empty()) {
            arma::uword last_entry = E.back();
            for (graph_node *vertex : current_node->neighbors) {
                if (!vertex->visited) {
                    found = true;
                    vertex->visited = true;
                    new_vertex = vertex;
                    new_entry = vertex->value;
                    break;
                }
            }
            if (!found) {
                auto last = E.back();
                E_bin[last] = 0;
                E.pop_back();
                current_node = current_node->parent;
                for (int i = 0; i < ndims; i++) {
                    counts[i][long_indices[last_entry][i]]--;
                }
            }
        }
        if (found) {
            E.push_back(new_entry);
            E_bin[new_entry] = 1;
            current_node = new_vertex;
            for (int i = 0; i < ndims; i++) {
                counts[i][long_indices[new_entry][i]]++;
            }
        }
    } while(!E.empty());
    cout << "iterations: " << iterations << endl;
    cout << endl;
    for (int i = 0; i <= nentries; i++) {
        int ntensors = binom(nentries, i);
        cout << ncompletable[i];
        cout << "/";
        cout << ntensors;
        cout << " of ";
        cout << dims_string;
        cout << " tensors with ";
        cout << i;
        cout << " observed entries are finitely completable" << endl;
        cout << flush;
    }
    return 0;
}

