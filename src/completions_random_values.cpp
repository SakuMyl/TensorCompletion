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
    bool latex_output = false;
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
    print_progress(0);
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
    vector<int> ncompletable(nentries + 1, 0);
    int n_tensors = 0;
    for (int i = 0; i <= nentries; i++) {
        n_tensors += binom(nentries, i);
    }
    int current_size = 1;
    vector<arma::uword> locations = { 0 };
    int progress = 1;
    int percentage = 0;
    int bar_width = 70;
    while (current_size != 0) {
        int n = current_size - 1;
        bool completable = is_finitely_completable(J, locations, Jrank);
        if (completable) {
            int entries_left = nentries - locations[locations.size() - 1] - 1;
            for (int i = current_size; i <= nentries; i++) {
                int n_completable_to_add = binom(entries_left, i - current_size);
                ncompletable[i] += n_completable_to_add;
                progress += n_completable_to_add;
            }
            if (locations[n] == nentries - 1) {
                current_size--;
                locations.pop_back();
                if (n > 0) {
                    locations[n - 1]++;
                }
            } else {
                locations[n]++;
            }
        } else {
            progress++;
            if (locations[n] == nentries - 1) {
                current_size--;
                locations.pop_back();
                if (n > 0) {
                    locations[n - 1]++;
                }
            } else {
                locations.push_back(locations[n] + 1);
                current_size++;
            }
        }
        float fprogress = (float) progress / n_tensors;
        int new_percentage = int(fprogress * 100);
        if (new_percentage > percentage) {
            percentage = new_percentage;
            print_progress(fprogress);
        }
    }
    cout << endl;
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

