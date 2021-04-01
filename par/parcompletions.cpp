#include <vector>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <random>
#include <limits>
#include <armadillo>
#include <iterator>

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
vector<T> *subsets_rec(
    vector<T> &in,
    vector<T> *out,
    vector<T> &current,
    int k,
    int index,
    int depth
    ) {
    if (depth == k) {
        vector<T> vec(current.size());
        copy(current.begin(), current.end(), vec.begin());
        *out = vec;
        return out + 1;
    }
    for (int i = index; i < in.size(); i++) {
        current.push_back(in[i]);
        out = subsets_rec(in, out, current, k, i + 1, depth + 1);
        current.pop_back();
    }
    return out;
}

template <typename T>
void subsets(
    vector<T> &in,
    vector<T> *out,
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
    std::string dims_string;
    for (int i = 0; i < dims.size(); i++) {
        dims_string += to_string(dims[i]);
        if (i != dims.size() - 1) dims_string += 'x';
    }
    if (dims.size() == 1) dims_string += "x1";
    cout << "Computing completability of " << dims_string << " partial tensors" << endl;
    //print_progress(0);
    int ndims = dims.size();
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

    int threads = omp_get_max_threads();
    vector<vector<int>> ncompletable(nentries + 1);
    for (int i = 0; i <= nentries; i++) {
        vector<int> vec(threads, 0);
        ncompletable[i] = vec;
    }
    vector<vector<arma::uword>> S(n_tensors);
    vector<arma::uword> *ptr = S.data();
    for (int i = 0; i <= nentries; i++) {
        subsets(all_indices, ptr, i);
        ptr += binom(nentries, i);
    }
    #pragma omp for schedule(dynamic, 16)
    for (int i = 0; i < S.size(); i++) {
        vector<arma::uword> &s = S[i];
        int id = omp_get_thread_num();
        if (is_finitely_completable(J, s, Jrank)) ncompletable[s.size()][id]++;
    }
    cout << endl;

    // for (int i = 1; i <= nentries; i++) {
    //     vector<vector<arma::uword>> S;
    //     subsets(all_indices, S, i);
    //     for (auto &s : S) {
    //         if (is_finitely_completable(J, s, Jrank)) ncompletable[i]++;
    //     }
    // }

    for (int i = 0; i <= nentries; i++) {
        int ntensors = binom(nentries, i);
        int ncompletable_all = 0;
        for (int j = 0; j < threads; j++) {
            ncompletable_all += ncompletable[i][j];
        }
        cout << ncompletable_all;
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

