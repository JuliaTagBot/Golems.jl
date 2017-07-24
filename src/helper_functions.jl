function symQuad(x::AbstractArray{<: Real,1}, S::AbstractArray{<: Real,2})
  out = x[1]^2 * S[1,1]
  for i ∈ 2:length(x)
    out += x[i]^2 * S[i,i]
    for j ∈ 1:i-1
      out += 2x[i] * x[j] * S[j,i]
    end
  end
  out
end


function traceProd(A::AbstractArray{<:Real,2}, B::AbstractArray{<:Real,2})
  sum(A .* B)
end
function traceSymProd(A::AbstractArray{<:Real,2}, B::AbstractArray{<:Real,2})
  out = A[1] * B[1]
  for i ∈ 2:size(A,2)
    out += A[i,i] * B[i,i]
    for j ∈ 1:i-1
      out += 2A[j,i] * B[j,i]
    end
  end
  out
end
function htraceSymProd(A::AbstractArray{<:Real,2}, B::AbstractArray{<:Real,2})
  out = A[1] * B[1] / 2
  for i ∈ 2:size(A,2)
    out += A[i,i] * B[i,i] / 2
    for j ∈ 1:i-1
      out += A[j,i] * B[j,i]
    end
  end
  out
end



function chol!(U::AbstractArray{<:Real,2}, Σ::AbstractArray{<:Real,2})
  for i ∈ 1:size(U,1)
    U[i,i] = Σ[i,i]
    for j ∈ 1:i-1
      U[j,i] = Σ[j,i]
      for k ∈ 1:j-1
        U[j,i] -= U[k,i] * U[k,j]
      end
      U[j,i] /= U[j,j]
      U[i,i] -= U[j,i]^2
    end
    U[i,i] = √U[i,i]
  end
end
function chol!(U::AbstractArray{<:Real,2})
  for i ∈ 1:size(U,1)
    for j ∈ 1:i-1
      for k ∈ 1:j-1
        U[j,i] -= U[k,i] * U[k,j]
      end
      U[j,i] /= U[j,j]
      U[i,i] -= U[j,i]^2
    end
    U[i,i] = √U[i,i]
  end
end
function inv!(U_inverse::AbstractArray{<:Real,2}, U::AbstractArray{<:Real,2})
  for i ∈ 1:size(U,1)
    U_inverse[i,i] = 1 / U[i,i]
    for j ∈ i+1:size(U,1)
      U_inverse[i,j] = U[i,j] * U_inverse[i,i]
      for k ∈ i+1:j-1
        U_inverse[i,j] += U[k,j] * U_inverse[i,k]
      end
      U_inverse[i,j] /= -U[j,j]
    end
  end
end

function inv!(U::AbstractArray{<:Real,2})
  for i ∈ 1:size(U,1)
    U[i,i] = 1 / U[i,i]
    for j ∈ i+1:size(U,1)
      U[i,j] = U[i,j] * U[i,i]
      for k ∈ i+1:j-1
        U[i,j] += U[k,j] * U[i,k]
      end
      U[i,j] /= -U[j,j]
    end
  end
end
function UUtij!(Σ::AbstractArray{<:Real,2}, U::AbstractArray{<:Real,2}, i::Int, j::Int)
  Σ[j,i] = U[1,i] * U[1,j]
  for k ∈ 2:j
    Σ[j,i] += U[k,i] * U[k,j]
  end
  Σ[i,j] = Σ[j,i]
end
function UUt!(Σ::AbstractArray{<:Real,2}, U::AbstractArray{<:Real,2})
  for i ∈ 1:size(Σ,1), j ∈ 1:i
    calc_Σij!(Σ, U, i, j)
  end
end
function UUtij!(Σ::AbstractArray{<:Real,2}, i::Int, j::Int)
  Σ[j,i] = Σ[i,i] * Σ[j,i]
  for k ∈ i+1:size(Σ,2)
    Σ[j,i] += Σ[i,k] * Σ[j,k]
  end
  Σ[i,j] = Σ[j,i]
end
function UUt!(Σ::AbstractArray{<:Real,2})
  for i ∈ 1:size(Σ,1), j ∈ 1:i
    UUtij!(Σ, i, j)
  end
end
function invSym!(S::AbstractArray{<:Real,2})
  chol!(S)
  inv!(S)
  UUt!(S)
  S
end
function triangleDet(A::AbstractArray{<:Real,2})
  out = A[1]
  for i ∈ 2:size(A,1)
    out *= A[i,i]
  end
  out
end
function logTriangleDet(A::AbstractArray{<:Real,2})
  log(triangleDet(A))
end

function eigenVJ(n::Int64)
  P = zeros(Float64, n, n)
  P[:,1] .= 1/√n
  for i ∈ 2:n
    ni = n - i + 1
    P[i:end,i] .= -1/√(ni*(ni+1))
    P[i-1,i] = √(ni/(ni+1))
  end
  P
end

logit(x::Real) = log( x / (1 - x) )
logistic(x::Real) = 1 / ( 1 + exp( - x ) )
sigmoid(x::Real) = log( (1 + x)/(1 - x) )
