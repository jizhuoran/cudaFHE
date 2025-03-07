#pragma once

namespace cudaFHE {

#define BARRET_PARAMS_0
#define BARRET_PARAMS_1 , const uint64_t *barret_mu

#define BARRET_ARGS_0
#define BARRET_ARGS_1 , barret_mu

#define GENERATE_TEMPLATE(NAME, HAS_BARRET)                                    \
  void NAME##_template(uint64_t *c, const uint64_t *a, const uint64_t *b,      \
                       const uint64_t *mod BARRET_PARAMS_##HAS_BARRET,         \
                       int64_t N, int64_t cur_limbs);

GENERATE_TEMPLATE(vadd, 0)
GENERATE_TEMPLATE(vsub, 0)
GENERATE_TEMPLATE(vmul, 1)
GENERATE_TEMPLATE(vadd_scalar, 0)
GENERATE_TEMPLATE(vsub_scalar, 0)
GENERATE_TEMPLATE(vmul_scalar, 1)
GENERATE_TEMPLATE(vneg, 0)

#undef BARRET_PARAMS_0
#undef BARRET_PARAMS_1
#undef BARRET_ARGS_0
#undef BARRET_ARGS_1
#undef GENERATE_TEMPLATE

void automorphism_transform_template(uint64_t *out_ptr, const uint64_t *in_ptr,
                                     int64_t l, int64_t N,
                                     const int32_t *precomp_vec_ptr);

void drop_last_element_scale_template(
    uint64_t *from_ptr, uint64_t *workspace_ptr, int64_t curr_limbs, int64_t l,
    int64_t L, int64_t N, const uint64_t *param_primes_ptr,
    const uint64_t *param_barret_ratio_ptr, const uint64_t *param_barret_k_ptr,
    const uint64_t *param_power_of_roots_shoup_ptr,
    const uint64_t *param_power_of_roots_ptr,
    const uint64_t *inverse_power_of_roots_div_two_ptr,
    const uint64_t *inverse_scaled_power_of_roots_div_two_ptr,
    const uint64_t *qlql_inv_mod_ql_div_ql_mod_q_ptr,
    const uint64_t *qlql_inv_mod_ql_div_ql_mod_q_shoup_ptr,
    const uint64_t *q_inv_mod_q_ptr, const uint64_t *q_inv_mod_q_shoup_ptr,
    uint64_t *to_ptr);

void innerproduct_template(const uint64_t *in_ptr, const uint64_t *bx_ptr,
                           const uint64_t *ax_ptr, int64_t curr_limbs,
                           int64_t L, int64_t N, int64_t sizeQP,
                           const uint64_t *primes_ptr,
                           const uint64_t *barret_ratio_ptr,
                           const uint64_t *barret_k_ptr, uint64_t *out_bx_ptr,
                           uint64_t *out_ax_ptr);

void moddown_cuda_template(
    uint64_t *to_ptr, uint64_t *workspace_ptr, uint64_t *from_ptr,
    int64_t curr_limbs, int64_t L, int64_t N, int64_t log_degree,
    const uint64_t *hat_inverse_vec_moddown_ptr,
    const uint64_t *hat_inverse_vec_shoup_moddown_ptr,
    const uint64_t *prod_q_i_mod_q_j_moddown_ptr,
    const uint64_t *prod_inv_moddown_ptr,
    const uint64_t *prod_inv_shoup_moddown_ptr, const uint64_t *primes_ptr,
    const uint64_t *barret_ratio_ptr, const uint64_t *barret_k_ptr,
    const uint64_t *power_of_roots_shoup_ptr,
    const uint64_t *power_of_roots_ptr,
    const uint64_t *inverse_power_of_roots_div_two_ptr,
    const uint64_t *inverse_scaled_power_of_roots_div_two_ptr);

void mod_raise_template(
    uint64_t *res_ptr, uint64_t *in_ptr, const uint64_t *moduliQ_ptr, int64_t N,
    int64_t L0, int64_t logN, int64_t level,
    const uint64_t *inverse_power_of_roots_div_two_ptr,
    const uint64_t *inverse_scaled_power_of_roots_div_two_ptr,
    const uint64_t *param_power_of_roots_shoup_ptr,
    const uint64_t *param_power_of_roots_ptr, const uint64_t *barret_ratio_ptr,
    const uint64_t *barret_k_ptr);

void modup_cuda_template(
    uint64_t *out_ptr, uint64_t *in_ptr, int64_t curr_limbs, int64_t L,
    const uint64_t *hat_inverse_vecs_ptr, int64_t hat_inverse_vecs_offset,
    const uint64_t *hat_inverse_vec_shoups_ptr,
    int64_t hat_inverse_vec_shoups_offset,
    const uint64_t *prod_q_i_mod_q_js_ptr, int64_t prod_q_i_mod_q_j_offset,
    const uint64_t *primes, const uint64_t *barret_ratio,
    const uint64_t *barret_k, int64_t N, int64_t sizeQP,
    const uint64_t *inverse_power_of_roots_div_two,
    const uint64_t *inverse_scaled_power_of_roots_div_two,
    const uint64_t *power_of_roots_shoup, const uint64_t *power_of_roots);

void mul_by_monomial_inplace_template(
    uint64_t *res_ptr, uint64_t *temp_ptr, const uint64_t *param_primes_ptr,
    int64_t l, int64_t N, int64_t M, int64_t monomialDeg, int64_t level,
    const uint64_t *inverse_power_of_roots_div_two_ptr,
    const uint64_t *inverse_scaled_power_of_roots_div_two_ptr,
    const uint64_t *param_power_of_roots_shoup_ptr,
    const uint64_t *param_power_of_roots_ptr);

} // namespace cudaFHE