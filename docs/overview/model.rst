Model and data conventions
==========================

The phenotype model is

.. math::

   y = C\gamma + X\beta + \epsilon,

where ``X`` is raw diploid dosage and ``C`` contains the fixed effects,
including an intercept. The evolutionary prior supplies a frequency-dependent
variance for each focal-trait effect ``beta_j``. The implementation uses

.. math::

   q_j = \hat{x}_j(1-\hat{x}_j),

with ``x_hat_j`` the sample allele frequency.

Simplified evolutionary model
-----------------------------

The simplified model fixes ``rho_ab^2 = 1`` and estimates ``sigma_b^2`` and
``tau``:

.. math::

   v_j = \frac{\sigma_b^2}{1 + 2\tau q_j}.

Full evolutionary model
-----------------------

The full model estimates the coupling in addition to the two scale parameters:

.. math::

   v_j = \sigma_b^2\left(1 - \rho_{ab}^2
   \frac{2\tau q_j}{1+2\tau q_j}\right).

The parameter constraints are ``sigma_b2 > 0``, ``tau >= 0``, and
``0 <= rho_ab^2 <= 1``. The simplified model is the exact nested
``rho_ab^2 = 1`` boundary of the full model, not a free reparameterization.

Variance components are estimated by profiled AI-REML. Dense inputs provide
an exact reference path; GRG inputs use matrix-free projected solves and
Hutchinson trace estimates with fixed seeded probes.
