// Exact maximum-area complete biclique by canonical closure enumeration.
// No aspect-ratio constraint. Minimum five vertices on each side, as in the notebook.
// Compile once for preparation: c++ -O3 -std=c++17 rectangle_search.cpp -o selector
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

using Bits = std::vector<uint64_t>;
using Clock = std::chrono::steady_clock;
int ns, ne, words, best = 0;
uint64_t nodes = 0;
std::vector<Bits> adjacency;
std::vector<int> order, best_stations;
std::vector<std::vector<uint8_t>> compatible;
Bits best_events;
Clock::time_point started;
double limit_seconds;

int count(const Bits& x) {
    int n = 0;
    for (auto word : x) n += __builtin_popcountll(word);
    return n;
}
Bits intersect(const Bits& x, const Bits& y) {
    Bits out(words);
    for (int w = 0; w < words; ++w) out[w] = x[w] & y[w];
    return out;
}
bool contains(const Bits& larger, const Bits& smaller) {
    for (int w = 0; w < words; ++w)
        if (smaller[w] & ~larger[w]) return false;
    return true;
}
int overlap(const Bits& x, const Bits& y) {
    int n = 0;
    for (size_t w = 0; w < x.size(); ++w) n += __builtin_popcountll(x[w] & y[w]);
    return n;
}
void update(const std::vector<int>& stations, const Bits& events) {
    int area = int(stations.size()) * count(events);
    if (stations.size() >= 5 && count(events) >= 5 && area > best) {
        best = area;
        best_stations.clear();
        for (int i : stations) best_stations.push_back(order[i]);
        best_events = events;
        std::cerr << "candidate=" << best << " stations=" << stations.size()
                  << " events=" << count(events) << " nodes=" << nodes << '\n';
    }
}
bool can_improve(const Bits& row, const std::vector<Bits>& rows) {
    std::vector<int> degrees;
    for (const auto& other : rows) {
        int degree = overlap(row, other);
        if (degree >= 5) degrees.push_back(degree);
    }
    std::sort(degrees.begin(), degrees.end(), std::greater<int>());
    for (int k = 5; k <= int(degrees.size()); ++k)
        if (k * degrees[k - 1] > best) return true;
    return false;
}
// A rectangle containing a vertex and k vertices on that same side has at most
// the kth largest pairwise-overlap number of opposite vertices. Remove vertices
// whose resulting area upper bound cannot beat the already saved incumbent.
void reduce(Bits& active_events) {
    bool changed = true;
    while (changed && ns > 0) {
        changed = false;
        std::vector<Bits> kept;
        std::vector<int> kept_order;
        for (int i = 0; i < ns; ++i) if (can_improve(adjacency[i], adjacency)) {
            kept.push_back(adjacency[i]); kept_order.push_back(order[i]);
        }
        if (int(kept.size()) < ns) changed = true;
        adjacency = kept; order = kept_order; ns = int(kept.size());
        std::vector<int> event_indices;
        std::vector<Bits> transpose;
        for (int e = 0; e < ne; ++e) if ((active_events[e / 64] >> (e % 64)) & 1) {
            Bits row((ns + 63) / 64, 0);
            for (int s = 0; s < ns; ++s)
                if ((adjacency[s][e / 64] >> (e % 64)) & 1) row[s / 64] |= uint64_t(1) << (s % 64);
            event_indices.push_back(e); transpose.push_back(row);
        }
        for (size_t i = 0; i < transpose.size(); ++i) if (!can_improve(transpose[i], transpose)) {
            int e = event_indices[i];
            active_events[e / 64] &= ~(uint64_t(1) << (e % 64)); changed = true;
        }
        for (auto& row : adjacency)
            for (int w = 0; w < words; ++w) row[w] &= active_events[w];
        std::cerr << "reduced graph=" << ns << "x" << count(active_events) << '\n';
    }
}
void visit(const std::vector<uint8_t>& intent, const Bits& extent, int start) {
    ++nodes;
    if (nodes % 100000 == 0)
        std::cerr << "search nodes=" << nodes << " best=" << best << " seconds="
                  << std::chrono::duration<double>(Clock::now() - started).count() << '\n';
    if (nodes % 1024 == 0 && std::chrono::duration<double>(Clock::now() - started).count() > limit_seconds)
        throw std::runtime_error("Exact selection timed out; no benchmark was frozen. Increase --search-seconds.");
    for (int i = start; i < ns; ++i) {
        if (intent[i]) continue;
        bool allowed = true;
        for (int j = 0; j < ns; ++j)
            if (intent[j] && !compatible[i][j]) { allowed = false; break; }
        if (!allowed) continue;
        Bits next = intersect(extent, adjacency[i]);
        int n_events = count(next);
        if (n_events < 5) continue;
        // CbO canonicity: an earlier new closure member was enumerated elsewhere.
        bool canonical = true;
        for (int j = 0; j < i; ++j) {
            if (!intent[j] && contains(adjacency[j], next)) { canonical = false; break; }
        }
        if (!canonical) continue;
        auto closure = intent;
        std::vector<int> selected;
        std::vector<std::pair<int, int>> candidates;
        for (int j = 0; j < ns; ++j) {
            if (intent[j] || (j >= i && contains(adjacency[j], next))) {
                closure[j] = 1;
                selected.push_back(j);
            } else if (j > i) {
                int degree = overlap(next, adjacency[j]);
                if (degree >= 5) candidates.emplace_back(degree, j);
            }
        }
        update(selected, next);
        allowed = true;
        for (int a : selected) for (int b : selected)
            if (a != b && !compatible[a][b]) allowed = false;
        if (!allowed) continue;
        // Compatible station pairs are necessary for any improving rectangle.
        // A proper coloring bounds a clique to one station per color. Max degree
        // within each color supplies the same safe kth-overlap area bound.
        std::sort(candidates.rbegin(), candidates.rend());
        std::vector<std::pair<int, int>> eligible;
        std::vector<Bits> candidate_events;
        for (auto candidate : candidates) {
            bool valid = true;
            for (int s : selected) if (!compatible[candidate.second][s]) { valid = false; break; }
            if (valid) {
                eligible.push_back(candidate);
                candidate_events.push_back(intersect(next, adjacency[candidate.second]));
            }
        }
        int required_events = 5;
        int max_stations = int(selected.size() + eligible.size());
        bool possible = false;
        while (max_stations >= 5) {
            std::vector<std::vector<int>> colors;
            std::vector<int> degrees;
            for (int v = 0; v < int(eligible.size()); ++v) {
                if (eligible[v].first < required_events) continue;
                int j = eligible[v].second;
                size_t c = 0;
                for (; c < colors.size(); ++c) {
                    bool independent = true;
                    for (int other : colors[c]) {
                        if (compatible[j][eligible[other].second] &&
                            overlap(candidate_events[v], candidate_events[other]) >= required_events) {
                            independent = false; break;
                        }
                    }
                    if (independent) break;
                }
                if (c == colors.size()) { colors.push_back({}); degrees.push_back(eligible[v].first); }
                colors[c].push_back(v);
            }
            std::sort(degrees.begin(), degrees.end(), std::greater<int>());
            int possible_stations = 0;
            for (int k = 1; k <= int(degrees.size()); ++k) {
                int stations = int(selected.size()) + k;
                if (stations >= 5 && stations * degrees[k - 1] > best)
                    possible_stations = stations;
            }
            max_stations = std::min(max_stations, possible_stations);
            if (max_stations < 5) break;
            // Any improving rectangle with <= max_stations stations needs this
            // many events. Recolor using that stronger necessary pair overlap.
            int tightened = std::max(5, best / max_stations + 1);
            if (tightened <= required_events) { possible = true; break; }
            required_events = tightened;
        }
        if (possible) visit(closure, next, i + 1);
    }
}
int main(int argc, char** argv) {
    limit_seconds = argc > 1 ? std::stod(argv[1]) : 3600;
    started = Clock::now();
    int edges;
    std::cin >> ns >> ne >> edges;
    words = (ne + 63) / 64;
    adjacency.assign(ns, Bits(words, 0));
    for (int k = 0, s, e; k < edges; ++k) {
        std::cin >> s >> e;
        adjacency.at(s).at(e / 64) |= uint64_t(1) << (e % 64);
    }
    order.resize(ns);
    std::iota(order.begin(), order.end(), 0);
    auto original = adjacency;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return count(original[a]) > count(original[b]); });
    for (int i = 0; i < ns; ++i) adjacency[i] = original[order[i]];
    // A deterministic greedy lower bound speeds the exact enumeration.
    for (int seed = 0; seed < std::min(ns, 5); ++seed) {
        Bits common = adjacency[seed];
        std::vector<int> chosen{seed};
        std::vector<uint8_t> used(ns, 0); used[seed] = 1;
        while (true) {
            int next = -1, degree = 4;
            for (int j = 0; j < ns; ++j) if (!used[j]) {
                int d = overlap(common, adjacency[j]);
                if (d > degree) { degree = d; next = j; }
            }
            if (next < 0) break;
            chosen.push_back(next); used[next] = 1;
            common = intersect(common, adjacency[next]);
            update(chosen, common);
        }
    }
    Bits all(words, ~uint64_t(0));
    if (ne % 64) all.back() = (uint64_t(1) << (ne % 64)) - 1;
    reduce(all);
    compatible.assign(ns, std::vector<uint8_t>(ns, 0));
    int compatible_pairs = 0;
    for (int i = 0; i < ns; ++i) for (int j = i + 1; j < ns; ++j) {
        if (can_improve(intersect(adjacency[i], adjacency[j]), adjacency)) {
            compatible[i][j] = compatible[j][i] = 1; ++compatible_pairs;
        }
    }
    std::cerr << "compatible station pairs=" << compatible_pairs << '\n';
    try { visit(std::vector<uint8_t>(ns, 0), all, 0); }
    catch (const std::runtime_error& error) { std::cerr << error.what() << '\n'; return 2; }
    if (best == 0) { std::cerr << "No complete rectangle with at least 5 stations and 5 events\n"; return 3; }
    std::vector<int> selected;
    for (int i : best_stations) selected.push_back(i);
    std::sort(selected.begin(), selected.end());
    std::cout << "{\"status\":\"OPTIMAL\",\"area\":" << best << ",\"nodes\":" << nodes << ",\"stations\":[";
    for (size_t i = 0; i < selected.size(); ++i) std::cout << (i ? "," : "") << selected[i];
    std::cout << "],\"events\":[";
    bool first = true;
    for (int e = 0; e < ne; ++e) if ((best_events[e / 64] >> (e % 64)) & 1) {
        std::cout << (first ? "" : ",") << e; first = false;
    }
    std::cout << "],\"seconds\":" << std::chrono::duration<double>(Clock::now() - started).count() << "}\n";
}
