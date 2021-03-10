#include <vector>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <random>
#include <limits>
#include <armadillo>
#include <iterator>
#include <unordered_set>
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
        subsets_rec(in, out, current, k + 1, i + 1, depth + 1);
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

int main() {
    vector<int> dims;
    get_dimensions(dims); 
    validate_dimensions(dims);
    vector<int> dimprods(dims.size() - 1);
    int mult = dims[dims.size() - 1];
    for (int i = dims.size() - 1; i > 0; i--) {
        dimprods[i - 1] = mult;
        cout << dimprods[i - 1] << endl;
        mult *= dims[i - 1];
    }
    std::string dims_string;
    for (int i = 0; i < dims.size(); i++) {
        dims_string += to_string(dims[i]);
        if (i != dims.size() - 1) dims_string += 'x';
    }
    if (dims.size() == 1) dims_string += "x1";
    cout << "Computing completability of " << dims_string << " partial tensors" << endl;
    print_progress(0);
    int ndims = dims.size();
    int nentries = product(dims);
    boost::dynamic_bitset<> b(nentries);
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
    vector<int> ncompletable(nentries, 0);
    int n_tensors = 0;
    for (int i = 0; i <= nentries; i++) {
        n_tensors += binom(nentries, i);
    }

    vector<vector<int>> dims_indices(dims.size());
    for (int i = 0; i < dims.size(); i++) {
        vector<int> vec(dims[i]);
        for (int j = 0; j < dims[i]; j++) {
            vec[j] = j;
        }
        dims_indices[i] = vec;
    }
    int tensors_in_total = 0;
    for (int i = 0; i <= nentries; i++) tensors_in_total += binom(nentries, i);
    vector<boost::dynamic_bitset<>> S(tensors_in_total);
    for (int i = 0, i < tensors_in_total; i++) {
        boost::dynamic_bitset s(nentries, i);
        S[i] = s;
    }
    vector<vector<int>> D;
    cart_product(dims_indices, D);
    for (int i = 1; i <= nentries; i++) {
        int ntensors = binom(nentries, i);
        vector<int> perm(dims[0]);
        iota(perm.begin(), perm.end(), 0);
        int n_completable = 0;
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

