#include <vector>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <random>
#include <limits>
#include <armadillo>
#include <iterator>
#include <unordered_map>
#include <boost/dynamic_bitset.hpp>

using namespace std;
using namespace arma;
const double EPS = 1E-9;

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

template <typename T>
void subsets_rec(
    vector<T> &in,
    vector<vector<T>> &out,
    vector<T> &current,
    int k,
    int index,
    int depth
    ) {
    if (depth == k) {
        out.push_back(current);
        return;
    }
    for (int i = index; i < in.size(); i++) {
        current.push_back(in[i]);
        subsets_rec(in, out, current, k, i + 1, depth + 1);
        current.pop_back();
    }
}

template <typename T>
void subsets(
    vector<T> &in,
    vector<vector<T>> &out,
    int k
    ) {
    assert(k >= 0);
    assert(k <= in.size());
    vector<T> current;
    subsets_rec(in, out, current, k, 0, 0);
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

struct vecHash
{
    size_t operator()(const vector<arma::uword> &V) const {
        int hash = V.size();
        for (auto &i : V) {
            hash ^= i + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        return hash;
    }
};

int index_vec_to_int(vector<arma::uword> &long_index, vector<int> &multipliers)
{
    int ret = 0;
    for (int i = 0; i < long_index.size(); i++) {
        ret += long_index[i] * multipliers[i];
    }
    return ret;
}

void shorten_indices(vector<vector<arma::uword>> &tensor, vector<arma::uword> &out, vector<int> &multipliers)
{
    for (int j = 0; j < tensor.size(); j++) {
        out[j] = index_vec_to_int(tensor[j], multipliers);
    }
}


int main() {
    vector<int> dims;
    get_dimensions(dims); 
    validate_dimensions(dims);
    sort(dims.begin(), dims.end());
    int ndims = dims.size();
    vector<int> dim_prods(ndims);
    dim_prods[ndims - 1] = 1;
    for (int i = ndims - 1; i > 0; i--) {
        dim_prods[i - 1] = dim_prods[i] * dims[i];
    }
    std::string dims_string;
    for (int i = 0; i < dims.size(); i++) {
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
    vector<tuple<int, int>> A;
    int first_index = 0;
    int prev = 0;
    for (int i = 0; i <= ndims; i++) {
        int d = i == ndims ? 0 : dims[i];
        if (prev != d) {
            if (i - first_index > 1) {
                A.push_back(make_tuple(first_index, i - 1));
            }
            first_index = i;
        }
        prev = d;
    }
    vector<int> tauperm_indices(ndims, 0);
    for (int i = 0; i < ndims; i++) {
        int j = -1;
        for (int k = 0; k < A.size(); k++) {
            tuple<int, int> pair = A[k];
            if (get<0>(pair) <= i && get<1>(pair) >= i) {
                j = k;
                break;
            }
        }
        tauperm_indices[i] = j;
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
    vector<int> ncompletable(nentries, 0);
    int n_tensors = 0;
    for (int i = 0; i <= nentries; i++) {
        n_tensors += binom(nentries, i);
    }

    vector<vector<arma::uword>> dims_indices(ndims);
    for (int i = 0; i < ndims; i++) {
        vector<arma::uword> vec(dims[i]);
        for (int j = 0; j < dims[i]; j++) {
            vec[j] = j;
        }
        dims_indices[i] = vec;
    }
    int tensors_in_total = 0;
    for (int i = 0; i <= nentries; i++) tensors_in_total += binom(nentries, i);
    vector<vector<arma::uword>> D;
    cart_product(dims_indices, D);
    #pragma omp parallel for 
    for (int i = 1; i <= nentries; i++) {
        vector<vector<vector<arma::uword>>> S;
        subsets(D, S, i);
        unordered_map<vector<arma::uword>, vector<vector<arma::uword>>, vecHash> Shash;
        for (auto &s : S) {
            vector<arma::uword> vec(i);
            for (int j = 0; j < i; j++) {
                vec[j] = index_vec_to_int(s[j], dim_prods);
            }
            Shash.insert({vec, s});
        }
        int ntensors = binom(nentries, i);
        vector<int> perm(dims[0]);
        iota(perm.begin(), perm.end(), 0);
        int n_completable = 0;
        while (!Shash.empty()) {
            int n_unique_tensors = 1;
            vector<vector<arma::uword>> T = Shash.begin()->second;
            vector<arma::uword> Tkey = Shash.begin()->first;
            Shash.extract(Tkey);
            vector<vector<arma::uword>> Tcopy(i);
            copy(T.begin(), T.end(), Tcopy.begin());
            while (next_permutation(perm.begin(), perm.end())) {
                for (int j = 0; j < T.size(); j++) {
                    Tcopy[j][0] = perm[T[j][0]];
                }
                vector<arma::uword> key(Tcopy.size());
                shorten_indices(Tcopy, key, dim_prods);
                if (!Shash.extract(key).empty()) n_unique_tensors++;
            }
            if (is_finitely_completable(J, Tkey, Jrank)) n_completable += n_unique_tensors;
        }
        cout << n_completable;
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

