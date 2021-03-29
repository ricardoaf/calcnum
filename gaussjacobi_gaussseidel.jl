# Solução de Sistemas de Equações Lineares: Métodos iterativos
# Gauss-Jacobi e Gauss-Seidel
using LinearAlgebra, Plots, Printf

# Função para atualização do vetor x
###################################################################################################
cn_update_x(A, b, x, i) = (b[i] - A[i, :] ⋅ x + A[i, i] * x[i]) / A[i, i]
cn_gaussJacobi(A, b, x, x0, i) = cn_update_x(A, b, x0, i)
cn_gaussSeidel(A, b, x, x0, i) = cn_update_x(A, b, x, i)

# Função de erro
###################################################################################################
function error(x, x_)
    nx, nΔx = norm(x), norm(x - x_)
    if nx < eps(); return nΔx else; return nΔx / nx; end
end

# Função para solução de sistema de equações lineares por métodos iterativos
###################################################################################################
function cn_slIter(A, b, x0=zeros(length(b)); met=cn_gaussSeidel, tol=1e-8, maxit=50)
    # Matriz A deve ser quadrada
    n = length(b)
    @assert size(A) == (n, n)
    # Inicialização
    err, nite, x = tol+1, 0, copy(x0)
    x_hist, err_hist = zeros(n, maxit+1), zeros(maxit)
    x_hist[:, 1] = x
    # Procedimento iterativo
    while err > tol && nite < maxit
        nite += 1
        # Atualização dos valores das variáveis, x
        for i = 1:n; x[i] = met(A, b, x, x_hist[:, nite], i); end
        # Atualização do erro
        err = error(x, x_hist[:, nite])
        # Atualização dos históricos
        x_hist[:, nite+1], err_hist[nite] = x, err
    end
    # Retorna históricos com valores das variáveis e erros por iteração
    return x_hist[:, 1:nite+1], err_hist[1:nite]
end

# Função com os critérios de convergência
###################################################################################################
function cn_conv(A)
    abs_A, n = abs.(A), size(A, 1)
    # Critério das linhas e colunas
    row = vec(sum(abs_A, dims=2)) ./ diag(abs_A) .- 1
    col = vec(sum(abs_A, dims=1)) ./ diag(abs_A) .- 1
    # Critério de Sassenfeld
    sas = zeros(n); sas[1] = row[1]
    for i = 2:n
        sas[i] = (sum(abs_A[i, i+1:end]) + sas[1:i-1] ⋅ abs_A[i, 1:i-1]) / abs_A[i, i]
    end
    return maximum(row), maximum(col), maximum(sas)
end

# Exemplos
###################################################################################################
exemplo = 3
if exemplo == 1
    A = [3 -0.1 -0.2; 0.1 7 -0.3; 0.3 -0.2 10];
    b = [7.85, -19.3, 71.4]
    x0 = [0., 0, 0]; tol = 1e-4; maxit = 50
elseif exemplo == 2
    A = [2. 1; 1 -2]; b = [2., -2]
    x0 = [0., 0]; tol = 1e-2; maxit = 50
elseif exemplo == 3
    A = [10. 2 1; 1 5 1; 2 3 10]; b = [14., 11, 8]
    x0 = [0., 0, 0]; tol = 1e-2; maxit = 50
elseif exemplo == 4
    A = [10. 5 3 1; 0 10 1 -2; 4 8 2 -1; 2 1 1 2];
    b = [9., -5, 3, 12]
    x0 = [0., 0, 0, 0]; tol = 1e-5; maxit = 200
end

# Cálculo da solução e apresentação dos resultados
###################################################################################################
fmt(output) = round.(output, sigdigits=4)
println("\nA: ", fmt(A)); println("b: ", fmt(b))

row, col, sas = cn_conv(A)
println("\nCritérios de convergência:")
@printf("linha: %.2f, coluna: %.2f, Sassenfeld: %.2f\n", row, col, sas)

println("\n### Gauss-Jacobi ###")
xGJ_hist, errGJ_hist = cn_slIter(A, b, x0, met=cn_gaussJacobi, tol=tol, maxit=maxit)
println("$(length(errGJ_hist)) iterações")
println("x_GJ: ", fmt(xGJ_hist))
println("err_GJ: ", fmt(errGJ_hist))

println("\n### Gauss-Seidel ###")
xGS_hist, errGS_hist = cn_slIter(A, b, x0, met=cn_gaussSeidel, tol=tol, maxit=maxit)
println("$(length(errGS_hist)) iterações")
println("x_GS: ", fmt(xGS_hist))
println("err_GS: ", fmt(errGS_hist))

gr(title="Histórico de erros", xlabel="Iteração", ylabel="Erro", yaxis=:log)
plot(errGJ_hist, label="GJ");
plot!(errGS_hist, label="GS")
