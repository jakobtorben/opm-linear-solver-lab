
#ifndef IMPROVED_BICGSTAB_HH
#define IMPROVED_BICGSTAB_HH

#include <array>
#include <cmath>
#include <complex>
#include <iostream>
#include <memory>
#include <type_traits>
#include <vector>

#include <dune/common/exceptions.hh>
#include <dune/common/math.hh>
#include <dune/common/simd/io.hh>
#include <dune/common/simd/simd.hh>
#include <dune/common/std/type_traits.hh>
#include <dune/common/timer.hh>

#include <dune/istl/allocator.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/istlexception.hh>
#include <dune/istl/operators.hh>
#include <dune/istl/preconditioner.hh>
#include <dune/istl/scalarproducts.hh>
#include <dune/istl/solver.hh>
#include <dune/istl/solverregistry.hh>



namespace Dune {

  // Ronald Kriemanns BiCG-STAB implementation from Sumo
  //! \brief Bi-conjugate Gradient Stabilized (BiCG-STAB)
  template<class X>
  class ImprovedBiCGSTABSolver : public IterativeSolver<X,X> {
  public:
    using typename IterativeSolver<X,X>::domain_type;
    using typename IterativeSolver<X,X>::range_type;
    using typename IterativeSolver<X,X>::field_type;
    using typename IterativeSolver<X,X>::real_type;

    // copy base class constructors
    using IterativeSolver<X,X>::IterativeSolver;

    // don't shadow four-argument version of apply defined in the base class
    using IterativeSolver<X,X>::apply;

    /*!
       \brief Apply inverse operator.

       \copydoc InverseOperator::apply(X&,Y&,InverseOperatorResult&)

       \note Currently, the BiCGSTABSolver aborts when it detects a breakdown.
     */
    virtual void apply (X& x, X& b, InverseOperatorResult& res)
    {
      using std::abs;
      const Simd::Scalar<real_type> EPSILON=1e-80;
      using std::abs;
      double it;
      field_type rho, rho_new, alpha, beta, h, omega;
      real_type norm;

      //
      // get vectors and matrix
      //
      X& r=b;
      X p(x);
      X v(x);
      X t(x);
      X y(x);
      X rt(x);

      //
      // begin iteration
      //

      // r = r - Ax; rt = r
      Iteration<double> iteration(*this,res);
      _prec->pre(x,r);             // prepare preconditioner

      _op->applyscaleadd(-1,x,r);  // overwrite b with defect

      rt=r;

      norm = _sp->norm(r);
      if(iteration.step(0, norm)){
        _prec->post(x);
        return;
      }
      p=0;
      v=0;

      rho   = 1;
      alpha = 1;
      omega = 1;

      //
      // iteration
      //

      for (it = 0.5; it < _maxit; it+=.5)
      {
        //
        // preprocess, set vecsizes etc.
        //

        // rho_new = < rt , r >
        rho_new = _sp->dot(rt,r);

        // look if breakdown occurred
        if (Simd::allTrue(abs(rho) <= EPSILON))
          DUNE_THROW(SolverAbort,"breakdown in BiCGSTAB - rho "
                     << Simd::io(rho) << " <= EPSILON " << EPSILON
                     << " after " << it << " iterations");
        if (Simd::allTrue(abs(omega) <= EPSILON))
          DUNE_THROW(SolverAbort,"breakdown in BiCGSTAB - omega "
                     << Simd::io(omega) << " <= EPSILON " << EPSILON
                     << " after " << it << " iterations");


        if (it>1) {
          beta = ( rho_new / rho ) * ( alpha / omega );
          p.axpy(-omega,v); // p = r + beta (p - omega*v)
          p *= beta;
          p += r;
        }
        else
        {
          p = r;
        }

        // y = W^-1 * p
        y = 0;
        _prec->apply(y,p);           // apply preconditioner

        // v = A * y
        _op->apply(y,v);

        // alpha = rho_new / < rt, v >
        h = _sp->dot(rt,v);

        if ( Simd::allTrue(abs(h) < EPSILON) )
          DUNE_THROW(SolverAbort,"abs(h) < EPSILON in BiCGSTAB - abs(h) "
                     << Simd::io(abs(h)) << " < EPSILON " << EPSILON
                     << " after " << it << " iterations");

        alpha = rho_new / h;



        // r = r - alpha*v
        r.axpy(-alpha,v);

        //
        // test stop criteria
        //

        norm = _sp->norm(r);
        if(iteration.step(it, norm)){
          // apply first correction to x
          // x <- x + alpha y
          x.axpy(alpha,y);
          break;
        }

        it+=.5;

        // y = W^-1 * r
        y = 0;
        _prec->apply(y,r);

        // t = A * y
        _op->apply(y,t);

        // omega = < t, r > / < t, t >
        omega = _sp->dot(t,r)/_sp->dot(t,t);

        // apply first correction to x
        // x <- x + alpha y
        x.axpy(alpha,y);

        // apply second correction to x
        // x <- x + omega y
        x.axpy(omega,y);

        // r = s - omega*t (remember : r = s)
        r.axpy(-omega,t);

        rho = rho_new;

        //
        // test stop criteria
        //

        norm = _sp->norm(r);
        if(iteration.step(it, norm)){
          break;
        }
      } // end for

      _prec->post(x);                  // postprocess preconditioner
    }

  protected:
    using IterativeSolver<X,X>::_op;
    using IterativeSolver<X,X>::_prec;
    using IterativeSolver<X,X>::_sp;
    using IterativeSolver<X,X>::_reduction;
    using IterativeSolver<X,X>::_maxit;
    using IterativeSolver<X,X>::_verbose;
    template<class CountType>
    using Iteration = typename IterativeSolver<X,X>::template Iteration<CountType>;
  };
  // DUNE_REGISTER_ITERATIVE_SOLVER("bicgstabsolver", defaultIterativeSolverCreator<Dune::BiCGSTABSolver>());

}

  #endif