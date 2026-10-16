/* GRAIL-0 graphlet engine (C kernels, called through ctypes from grail0/graphlets.py).
 *
 * 1) esu_count: exact ESU enumeration (Wernicke 2006) of every connected induced
 *    subgraph with 3..kmax nodes (kmax <= 5). Each subset is visited exactly once; its
 *    induced adjacency bitmask (in ESU discovery order) is mapped to (graphlet type,
 *    orbit of every position) by lookup tables built in Python. Outputs per-graph type
 *    counts and per-node orbit counts (GDV).
 *    Bit convention: pair (i, j), i < j, of positions -> bit j*(j-1)/2 + i.
 *
 * 2) orbit4_noninduced: exact closed-form NON-induced node-level counts for the
 *    2..4-node orbits 0..14 (degree, P3, triangle, P4, 3-star, C4, paw, diamond, K4)
 *    in O(sum_v d_v^2 + #triangles*d + #K4) time. Python converts them to induced orbit
 *    counts with the (unitriangular) inclusion matrix derived by brute force from the
 *    same lookup tables. Used for the social graphs (hubs of degree ~3000), where ESU
 *    is infeasible. Cross-checked against esu_count in grail0/tests.py.
 *
 * Graphs are CSR: indptr int64[n+1], idx int32[nnz], symmetric, no self loops,
 * neighbours sorted ascending within each row.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int64_t esu_count(int64_t n, const int64_t *indptr, const int32_t *idx, int kmax,
                  const int16_t *tab_type, const int16_t *tab_orb,
                  int64_t *counts, int64_t *gdv, int n_orb)
{
    if (n <= 0) return 0;
    if (kmax < 3 || kmax > 5) return -2;
    int32_t *cnt = (int32_t *)calloc((size_t)n, sizeof(int32_t)); /* |closed nbhd hits| */
    int32_t *pos = (int32_t *)malloc((size_t)n * sizeof(int32_t)); /* position in subset */
    int32_t *ext = (int32_t *)malloc((size_t)kmax * (size_t)n * sizeof(int32_t));
    if (!cnt || !pos || !ext) { free(cnt); free(pos); free(ext); return -1; }
    for (int64_t i = 0; i < n; i++) pos[i] = -1;
    int64_t ext_len[8], ext_i[8];
    int32_t sub[8];
    int32_t mask[8];
    int64_t total = 0;
    for (int64_t v = 0; v < n; v++) {
        sub[0] = (int32_t)v; pos[v] = 0; mask[1] = 0;
        cnt[v]++;
        for (int64_t e = indptr[v]; e < indptr[v + 1]; e++) cnt[idx[e]]++;
        int64_t L = 0;
        for (int64_t e = indptr[v]; e < indptr[v + 1]; e++)
            if (idx[e] > v) ext[L++] = idx[e];
        ext_len[1] = L; ext_i[1] = 0;
        int depth = 1;
        while (depth >= 1) {
            if (depth < kmax && ext_i[depth] < ext_len[depth]) {
                const int32_t *cur = ext + (size_t)(depth - 1) * (size_t)n;
                int32_t w = cur[ext_i[depth]];
                ext_i[depth]++;
                int64_t L2 = 0;
                if (depth + 1 < kmax) {
                    int32_t *nxt = ext + (size_t)depth * (size_t)n;
                    for (int64_t j = ext_i[depth]; j < ext_len[depth]; j++) nxt[L2++] = cur[j];
                    for (int64_t e = indptr[w]; e < indptr[w + 1]; e++) {
                        int32_t u = idx[e];
                        if (u > v && cnt[u] == 0) nxt[L2++] = u; /* exclusive neighbours */
                    }
                }
                int p = depth;
                int32_t m = mask[depth];
                for (int64_t e = indptr[w]; e < indptr[w + 1]; e++) {
                    int32_t q = pos[idx[e]];
                    if (q >= 0) m |= (int32_t)1 << (p * (p - 1) / 2 + q);
                }
                sub[p] = w; pos[w] = p;
                cnt[w]++;
                for (int64_t e = indptr[w]; e < indptr[w + 1]; e++) cnt[idx[e]]++;
                depth++;
                mask[depth] = m;
                ext_len[depth] = L2; ext_i[depth] = 0;
                int s = depth;
                if (s >= 3) {
                    int t = tab_type[s * 1024 + m];
                    if (t < 0) { free(cnt); free(pos); free(ext); return -3; }
                    counts[t]++;
                    const int16_t *ob = tab_orb + ((size_t)(s * 1024 + m)) * 5;
                    for (int q = 0; q < s; q++) gdv[(size_t)sub[q] * (size_t)n_orb + ob[q]]++;
                    total++;
                }
            } else {
                int32_t w = sub[depth - 1];
                cnt[w]--;
                for (int64_t e = indptr[w]; e < indptr[w + 1]; e++) cnt[idx[e]]--;
                pos[w] = -1;
                depth--;
            }
        }
    }
    free(cnt); free(pos); free(ext);
    return total;
}

/* out: int64[n*15], column j = non-induced count of orbit j (j=0..14) at each node. */
int orbit4_noninduced(int64_t n, const int64_t *indptr, const int32_t *idx, int64_t *out)
{
    if (n <= 0) return 0;
    int64_t nnz = indptr[n];
    int64_t *deg = (int64_t *)malloc((size_t)n * sizeof(int64_t));
    int64_t *tE = (int64_t *)calloc((size_t)(nnz > 0 ? nnz : 1), sizeof(int64_t));
    int64_t *tV = (int64_t *)calloc((size_t)n, sizeof(int64_t));
    int64_t *W = (int64_t *)calloc((size_t)n, sizeof(int64_t));
    int64_t *mark = (int64_t *)calloc((size_t)n, sizeof(int64_t));   /* stamp / entry+1 */
    char *markV = (char *)calloc((size_t)n, 1);
    int64_t *cn = (int64_t *)calloc((size_t)n, sizeof(int64_t));
    int32_t *touched = (int32_t *)malloc((size_t)n * sizeof(int32_t));
    int32_t *L = (int32_t *)malloc((size_t)n * sizeof(int32_t));
    int64_t *rank = (int64_t *)malloc((size_t)n * sizeof(int64_t));
    if (!deg || !tE || !tV || !W || !mark || !markV || !cn || !touched || !L || !rank) {
        free(deg); free(tE); free(tV); free(W); free(mark); free(markV); free(cn);
        free(touched); free(L); free(rank); return -1;
    }
    for (int64_t v = 0; v < n; v++) deg[v] = indptr[v + 1] - indptr[v];
    /* rank: (deg, id) total order; orientation u->w iff key(w) > key(u) */
    for (int64_t v = 0; v < n; v++) rank[v] = deg[v] * (n + 1) + v;
    /* triangles per directed entry: tE[e(u,w)] = |N(u) & N(w)| */
    for (int64_t u = 0; u < n; u++) {
        for (int64_t e = indptr[u]; e < indptr[u + 1]; e++) mark[idx[e]] = u + 1;
        for (int64_t e = indptr[u]; e < indptr[u + 1]; e++) {
            int32_t w = idx[e];
            int64_t c = 0;
            for (int64_t f = indptr[w]; f < indptr[w + 1]; f++) if (mark[idx[f]] == u + 1) c++;
            tE[e] = c;
            tV[u] += c;
        }
    }
    for (int64_t u = 0; u < n; u++) { tV[u] /= 2; mark[u] = 0; }
    for (int64_t v = 0; v < n; v++) {
        int64_t s = 0;
        for (int64_t e = indptr[v]; e < indptr[v + 1]; e++) s += deg[idx[e]] - 1;
        W[v] = s;
    }
    for (int64_t v = 0; v < n; v++) {
        int64_t d = deg[v], t = tV[v];
        int64_t *o = out + (size_t)v * 15;
        o[0] = d;
        o[1] = W[v];
        o[2] = d * (d - 1) / 2;
        o[3] = t;
        int64_t sW = 0, s6 = 0, s9 = 0, s10 = 0, s13 = 0;
        for (int64_t e = indptr[v]; e < indptr[v + 1]; e++) {
            int32_t u = idx[e];
            int64_t du = deg[u];
            sW += W[u];
            s6 += (du - 1) * (du - 2) / 2;
            s9 += tV[u] - tE[e];
            s10 += tE[e] * (du - 2);
            s13 += tE[e] * (tE[e] - 1) / 2;
        }
        o[4] = sW - d * (d - 1) - 2 * t;
        o[5] = (d - 1) * W[v] - 2 * t;
        o[6] = s6;
        o[7] = d * (d - 1) * (d - 2) / 6;
        /* C4 through v: sum_{c != v} C(cn(v,c), 2) */
        int64_t nt = 0;
        for (int64_t e = indptr[v]; e < indptr[v + 1]; e++) {
            int32_t u = idx[e];
            for (int64_t f = indptr[u]; f < indptr[u + 1]; f++) {
                int32_t c = idx[f];
                if (c == v) continue;
                if (cn[c] == 0) touched[nt++] = c;
                cn[c]++;
            }
        }
        int64_t s8 = 0;
        for (int64_t i = 0; i < nt; i++) { int64_t k = cn[touched[i]]; s8 += k * (k - 1) / 2; cn[touched[i]] = 0; }
        o[8] = s8;
        o[9] = s9;
        o[10] = s10;
        o[11] = t * (d - 2);
        o[12] = 0;
        o[13] = s13;
        o[14] = 0;
    }
    /* oriented triangle + K4 enumeration for orbits 12 (diamond, deg-2 node) and 14 (K4) */
    for (int64_t u = 0; u < n; u++) {
        for (int64_t e = indptr[u]; e < indptr[u + 1]; e++) {
            int32_t w = idx[e];
            if (rank[w] > rank[u]) mark[w] = e + 1;
        }
        for (int64_t e1 = indptr[u]; e1 < indptr[u + 1]; e1++) {
            int32_t v = idx[e1];
            if (rank[v] <= rank[u]) continue;
            int64_t nl = 0;
            for (int64_t e2 = indptr[v]; e2 < indptr[v + 1]; e2++) {
                int32_t x = idx[e2];
                if (rank[x] <= rank[v]) continue;
                markV[x] = 1;
                if (mark[x]) {
                    /* triangle {u, v, x}, rank u < v < x */
                    int64_t t_uv = tE[e1], t_vx = tE[e2], t_ux = tE[mark[x] - 1];
                    out[(size_t)u * 15 + 12] += t_vx - 1;
                    out[(size_t)v * 15 + 12] += t_ux - 1;
                    out[(size_t)x * 15 + 12] += t_uv - 1;
                    L[nl++] = x;
                }
            }
            /* K4 {u, v, x, y}: x in L (common out-nbrs of u, v), y out-nbr of x, common to u, v */
            for (int64_t i = 0; i < nl; i++) {
                int32_t x = L[i];
                for (int64_t f = indptr[x]; f < indptr[x + 1]; f++) {
                    int32_t y = idx[f];
                    if (rank[y] <= rank[x]) continue;
                    if (mark[y] && markV[y]) {
                        out[(size_t)u * 15 + 14]++;
                        out[(size_t)v * 15 + 14]++;
                        out[(size_t)x * 15 + 14]++;
                        out[(size_t)y * 15 + 14]++;
                    }
                }
            }
            for (int64_t e2 = indptr[v]; e2 < indptr[v + 1]; e2++) markV[idx[e2]] = 0;
        }
        for (int64_t e = indptr[u]; e < indptr[u + 1]; e++) mark[idx[e]] = 0;
    }
    free(deg); free(tE); free(tV); free(W); free(mark); free(markV); free(cn);
    free(touched); free(L); free(rank);
    return 0;
}
