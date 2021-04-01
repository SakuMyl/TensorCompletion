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

int index_vec_to_int(vector<int> &long_index, vector<int> &multipliers)
{
    int ret = 0;
    for (int i = 0; i < long_index.size(); i++) {
        ret += long_index[i] * multipliers[i];
    }
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

template <typename T>
vector<T> not_in_place_sort(vector<T> in)
{
    sort(in.begin(), in.end());
    return in;
}

vector<vector<int>> get_canonical_form(vector<vector<int>> tensor, vector<vector<int>> counts, vector<int> &dims)
{
    vector<vector<int>> idx(counts.size());
    for (int i = 0; i < idx.size(); i++) {
        vector<int> vec(counts[i].size());
        iota(vec.begin(), vec.end(), 0);
        idx[i] = vec;
    }
    vector<int> idx_tr(counts.size());
    iota(idx_tr.begin(), idx_tr.end(), 0);
    // Sort each "dimension"
    for (int i = 0; i < counts.size(); i++) {
        auto &v = counts[i];
        sort(idx[i].begin(), idx[i].end(),
            [&v](int i1, int i2){ return v[i1] > v[i2]; }
        );
        sort(v.begin(), v.end());
    }
    // Sort by lexicographic ordering of vectors for dimensions of equal length
    int prev_index = 0;
    for (int i = 1; i <= dims.size(); i++) {
        if (i == dims.size() || dims[i] != dims[prev_index]) {
            sort(idx_tr.begin() + prev_index, idx_tr.begin() + i,
                [&counts](int i1, int i2){ return counts[i1] > counts[i2]; }
            );
            prev_index = i;
        }
    }
    vector<vector<int>> idx_inv = idx;
    for (int i = 0; i < idx.size(); i++) {
        auto &v = idx[i];
        auto &v_inv = idx_inv[i];
        for (int j = 0; j < idx[i].size(); j++) {
            v_inv[v[j]] = j;
        }
    }
    vector<int> idx_tr_inv = idx_tr;
    for (int i = 0; i < idx_tr.size(); i++) {
        idx_tr_inv[idx_tr[i]] = i;
    }
    for (auto &v : tensor) {
        for (int j = 0; j < dims.size(); j++) {
            int dim = idx_tr_inv[j];
            v[j] = idx_inv[dim][v[j]];
        }
    }
    sort(tensor.begin(), tensor.end());
    return tensor;
}

struct vecHash
{
    size_t operator()(const vector<vector<int>> &V) const {
        int hash = V.size();
        for (auto &v : V) {
            for (auto &i : v) {
                hash ^= i + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            }
        }
        return hash;
    }
};

void increment_index_and_change_counts(vector<vector<int>> &counts, vector<int> &index, vector<int> &dims)
{
    int i = dims.size() - 1;
    while (i > 0 && index[i] == dims[i] - 1) {
        index[i] = 0;
        counts[i][counts[i].size() - 1]--;
        counts[i][0]++;
        i--;
    }
    index[i]++;
    counts[i][index[i] - 1]--;
    counts[i][index[i]]++;
}

vector<int> add_index_and_change_counts(vector<vector<int>> &counts, vector<int> index, vector<int> &dims)
{
    int i = dims.size() - 1;
    while (i > 0 && index[i] == dims[i] - 1) {
        index[i] = 0;
        counts[i][0]++;
        i--;
    }
    index[i]++;
    for (int j = 0; j <= i; j++) {
        counts[j][index[j]]++;
    }
    return index;
}

int main() {
    bool latex_output = false;
    vector<int> dims;
    get_dimensions(dims); 
    validate_dimensions(dims);
    int ndims = dims.size();
    sort(dims.begin(), dims.end());

    std::string dims_string;
    for (int i = 0; i < ndims; i++) {
        dims_string += to_string(dims[i]);
        if (i != dims.size() - 1) dims_string += 'x';
    }
    if (ndims == 1) dims_string += "x1";
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
    vector<int> dim_prods(ndims);
    dim_prods[ndims - 1] = 1;
    for (int i = ndims - 1; i > 0; i--) {
        dim_prods[i - 1] = dim_prods[i] * dims[i];
    }
    vector<arma::uword> all_indices(nentries);
    iota(all_indices.begin(), all_indices.end(), 0);
    int Jrank = get_jacobian_rank(J, all_indices);
    vector<int> ncompletable(nentries + 1, 0);
    int n_tensors = 0;
    for (int i = 0; i <= nentries; i++) {
        n_tensors += binom(nentries, i);
    }
    int current_size = 1;
    int progress = 1;
    int percentage = 0;
    int bar_width = 70;
    vector<vector<int>> counts(ndims);
    for (int i = 0; i < ndims; i++) {
        vector<int> vec(dims[i], 0);
        vec[0] = 1;
        counts[i].assign(vec.begin(), vec.end());
    }
    unordered_map<vector<vector<int>>, bool, vecHash> representatives;
    vector<arma::uword> simple_locations = { 0 };
    vector<vector<int>> locations;
    vector<int> first(ndims, 0);
    locations.push_back(first);
    int n_orbits = 0;
    while (current_size != 0) {
        auto canonical_form = get_canonical_form(locations, counts, dims);
        auto search = representatives.find(canonical_form);
        bool new_orbit = search == representatives.end();
        if (new_orbit) {
            n_orbits++;
            bool completable = is_finitely_completable(J, simple_locations, Jrank);
            representatives.insert({ canonical_form, completable });
            if (completable) ncompletable[current_size]++;
        } else {
            if (search->second) ncompletable[current_size]++;
        }
        int n = current_size - 1;
        progress++;
        if (simple_locations[n] == nentries - 1) {
            current_size--;
            for (int i = 0; i < ndims; i++) {
                counts[i][locations[n][i]]--;
            }
            locations.pop_back();
            simple_locations.pop_back();
            if (n > 0) {
                simple_locations[n - 1]++;
                increment_index_and_change_counts(counts, locations[n - 1], dims);
            }
        } else {
            locations.push_back(add_index_and_change_counts(counts, locations[n], dims));
            simple_locations.push_back(simple_locations[n] + 1);
            current_size++;
        }
        float fprogress = (float) progress / n_tensors;
        int new_percentage = int(fprogress * 100);
        if (new_percentage > percentage) {
            percentage = new_percentage;
            //print_progress(fprogress);
        }
    }
    cout << endl;
    cout << "number of orbits: " << n_orbits << endl;
    if (latex_output) {
        cout << "Size & Finitely completable & Total \\\\" << endl;
        cout << "\\hline" << endl;
        for (int i = 0; i <= nentries; i++) {
            int ntensors = binom(nentries, i);
            cout << i << " & " << ncompletable[i] << " & " << ntensors << " \\\\" << endl;
            cout << "\\hline" << endl;
        }
    } else {
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
    }
    return 0;
}

