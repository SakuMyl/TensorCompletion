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

int index_vec_to_int(vector<arma::uword> &long_index, vector<int> &multipliers)
{
    int ret = 0;
    for (int i = 0; i < long_index.size(); i++) {
        ret += long_index[i] * multipliers[i];
    }
    return ret;
}

void int_to_index_vec(arma::uword in, vector<arma::uword> &out, vector<int> &multipliers, vector<int> &dims)
{
    for (int i = 0; i < multipliers.size(); i++) {
        out[i] = (in / multipliers[i]) % dims[i];
    }
}

void shorten_indices(vector<vector<arma::uword>> &tensor, vector<arma::uword> &out, vector<int> &multipliers)
{
    for (int j = 0; j < tensor.size(); j++) {
        out[j] = index_vec_to_int(tensor[j], multipliers);
    }
}

int get_orbit_size(vector<vector<int>> &E)
{
    // Mock for now
    return 1;
}

// Checks whether adding a given entry will change the counts so that the two rules are still satisfied
bool satisfies_rules(vector<vector<int>> &counts, vector<int> &dims, vector<arma::uword> &entry)
{
    vector<vector<int>> counts_cpy(counts.size());
    for (int i = 0; i < counts.size(); i++) {
        vector<int> vec(counts[i].size());
        counts_cpy[i] = vec;
        for (int j = 0; j < vec.size(); j++) {
            counts_cpy[i][j] = counts[i][j];
        }
    }
    for (int k = 0; k < entry.size(); k++) {
        counts_cpy[k][entry[k]]++;
    }
    // Rule 1
    for (int k = 0; k < entry.size(); k++) {
        for (int i = 1; i < dims[k]; i++) {
            if (counts_cpy[k][i - 1] < counts_cpy[k][i]) {
                return false;
            }
        }
        //arma::uword i = entry[k];
        //if (i > 0 && counts_cpy[k][i - 1] < counts_cpy[k][i]) return false; 
    }
    // Rules 2 and 3
    for (int k = 1; k < entry.size(); k++) {
        if (dims[k] != dims[k - 1]) continue;
        int i = 0;
        bool equal = true;
        while (equal && i < dims[k]) {
            int diff = counts_cpy[k - 1][i] - counts_cpy[k][i]; 
            if (diff < 0) {
                return false;
            }
            if (diff > 0) {
                equal = false;
            }
            i++;
        }
    }
    // for (int k = 1; k < entry.size(); k++) {
    //     arma::uword i = entry[k];
    //     int diff = counts[k - 1][i] - counts[k][i];
    //     arma::uword j = entry[k - 1];
    //     if (j == i) diff++;
    //     if (i == 0 && diff <= 0) return false;
    //     else if (i > 0 && diff <= 0) {
    //         int jdiff = counts[k - 1][i - 1] - counts[k][i - 1];
    //         if (j == i - 1) jdiff++;
    //         else if (j == i) diff++;
    //         if (jdiff == 0) return false;
    //     }
    // }
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
    //vector<arma::uword> vec(ndims, 0);
    //E[0] = vec;
    int M = 1;
    //flat_indices[0] = index_vec_to_int(E[0], dim_prods);
    //stack<vector<arma::uword>> entry_stack;

    // Adjacency list
    //int G[nentries][nentries];
    list<graph_node *> l;
    graph_node G = { l, nullptr, 0, true };
    graph_node *current_node = &G;
    // for (int i = 0; i < nentries; i++) {
    //     for (int j = 0; j < nentries; j++) {
    //         G[i][j] = 0;
    //     }
    // }
    // counts[i][j] is how many entries have the ith index equal to j
    vector<vector<int>> counts(ndims);
    for (int i = 0; i < ndims; i++) {
        vector<int> vec(dims[i], 0);
        vec[0] = 1;
        counts[i].assign(vec.begin(), vec.end());
    }
    int vertices_left = 1;
    int iterations = 0;
    int vertices_added = 1;
    do {
        //cout << "current_node value: " << current_node->value << endl;
        iterations++;
        int M = E.size();
        if (is_finitely_completable(J, E, Jrank)) {
            int n = get_orbit_size(counts);
            ncompletable[M] += n;
        }
        vector<arma::uword> new_entries;
        arma::uword last_added = E.back();
        vector<arma::uword> index_vec(ndims);
        int first_index = last_added / dim_prods[0];
        index_vec[0] = (last_added +  1) / dim_prods[0];
        // Loop through potential new entries to add satisfying the two rules
        for (arma::uword i = last_added + 1; index_vec[0] <= first_index + 1 && i < nentries; i++) {
            int_to_index_vec(i, index_vec, dim_prods, dims);
            bool is_neighbor = false;
            for (graph_node *vertex : current_node->neighbors) {
                if (vertex->value == i) {
                    is_neighbor = true;
                    break;
                }
            }
            if (!is_neighbor && satisfies_rules(counts, dims, index_vec)) {
                new_entries.push_back(i);
            }
            // if (G[last_added][i] == 0 && satisfies_rules(counts, dims, index_vec)) {
            //     new_entries.push_back(i);
            // }
        }
            // Mark visited
            //flat_indices.pop_back();
        // Add new vertices to visit
        vertices_left += new_entries.size();
        vertices_added += new_entries.size();
        //cout << "iteration: " << iterations << ", vertices added: " << new_entries.size() << endl;
        // for (int i = 0; i < new_entries.size(); i++) {
        //     //entry_stack.push(new_entries[i]);
        //     G[last_added][new_entries[i]] = 1;
        //     cout << "entry " << new_entries[i] << " added" << endl;
        // }
        for (int i = 0; i < new_entries.size(); i++) {
            list<graph_node *> l;
            graph_node *vertex = new graph_node();
            vertex->neighbors = l;
            vertex->parent = current_node;
            vertex->value = new_entries[i];
            vertex->visited = false;
            current_node->neighbors.push_back(vertex);
            //cout << "value of vertex added: " << vertex->value << endl;
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
            // for (int i = 0; i < nentries && !found; i++) {
            //     if (G[last_entry][i] == 1) {
            //         G[last_entry][i] = 2;
            //         vertices_left--;
            //         found = true;
            //         new_entry = i;
            //     }
            // }
            if (!found) {
                E.pop_back();
                current_node = current_node->parent;
                vector<arma::uword> index_vec(ndims);
                int_to_index_vec(last_entry, index_vec, dim_prods, dims);
                for (int i = 0; i < ndims; i++) {
                    counts[i][index_vec[i]]--;
                }
            }
        }
        if (found) {
            E.push_back(new_entry);
            current_node = new_vertex;
            vector<arma::uword> index_vec(ndims);
            int_to_index_vec(new_entry, index_vec, dim_prods, dims);
            for (int i = 0; i < ndims; i++) {
                counts[i][index_vec[i]]++;
            }
        }
        //flat_indices.push_back(index_vec_to_int(new_entry, dim_prods));
    } while(!E.empty());
    cout << "iterations: " << iterations << ", vertices: " << vertices_added << endl;
    cout << endl;
    // // for (int i = 0; i <= nentries; i++) {
    // //     int ntensors = binom(nentries, i);
    // //     cout << ncompletable[i - 1];
    // //     cout << "/";
    // //     cout << ntensors;
    // //     cout << " of ";
    // //     cout << dims_string;
    // //     cout << " tensors with ";
    // //     cout << i;
    // //     cout << " observed entries are finitely completable" << endl;
    // //     cout << flush;
    // // }
    return 0;
}

