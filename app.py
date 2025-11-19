from flask import Flask, render_template, request, redirect, url_for
import sympy as sp
import math

app = Flask(__name__)

# Helpers (similar to CLI)
def make_function(func_str):
    x = sp.symbols('x')
    expr = sp.sympify(func_str)
    f = sp.lambdify(x, expr, modules=["math"])
    return f, expr

def derivative_of(expr):
    x = sp.symbols('x')
    dexpr = sp.diff(expr, x)
    return sp.lambdify(x, dexpr, modules=["math"]), dexpr

# Methods (same logic but returns serializable rows)
def bisection(f, a, b, tol, max_iter):
    fa = f(a); fb = f(b)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs.")
    rows = []
    for k in range(1, max_iter+1):
        c = (a+b)/2
        fc = f(c)
        err = abs(b-a)/2
        rows.append({'k':k,'a':a,'b':b,'c':c,'fa':fa,'fb':fb,'fc':fc,'err':err})
        if abs(fc) < tol or err < tol:
            return rows, c, fc, err, k
        if fa * fc < 0:
            b = c; fb = fc
        else:
            a = c; fa = fc
    return rows, c, fc, err, k

def regula_falsi(f,a,b,tol,max_iter):
    fa = f(a); fb = f(b)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs.")
    rows = []
    c = a
    for k in range(1, max_iter+1):
        c_prev = c
        c = (a*fb - b*fa)/(fb - fa)
        fc = f(c)
        err = abs(c - c_prev) if k>1 else abs(b-a)
        rows.append({'k':k,'a':a,'b':b,'c':c,'fa':fa,'fb':fb,'fc':fc,'err':err})
        if abs(fc) < tol or err < tol:
            return rows, c, fc, err, k
        if fa * fc < 0:
            b = c; fb = fc
        else:
            a = c; fa = fc
    return rows, c, fc, err, k

def secant(f,x0,x1,tol,max_iter):
    rows=[]
    f0=f(x0); f1=f(x1)
    for k in range(1,max_iter+1):
        if (f1 - f0)==0:
            raise ZeroDivisionError("Zero denominator in Secant")
        x2 = x1 - f1*(x1-x0)/(f1-f0)
        f2 = f(x2)
        err = abs(x2 - x1)
        rows.append({'k':k,'x0':x0,'x1':x1,'x2':x2,'f0':f0,'f1':f1,'f2':f2,'err':err})
        if abs(f2) < tol or err < tol:
            return rows, x2, f2, err, k
        x0,f0 = x1,f1
        x1,f1 = x2,f2
    return rows, x2, f2, err, k

def newton_raphson(f, fprime_callable, x0, tol, max_iter):
    rows=[]
    x=x0; fx=f(x)
    for k in range(1,max_iter+1):
        fpx = fprime_callable(x)
        if fpx == 0:
            raise ZeroDivisionError("Derivative zero.")
        x_new = x - fx/fpx
        fx_new = f(x_new)
        err = abs(x_new - x)
        rows.append({'k':k,'x':x,'x_new':x_new,'fx':fx,'fpx':fpx,'fx_new':fx_new,'err':err})
        if abs(fx_new) < tol or err < tol:
            return rows, x_new, fx_new, err, k
        x,fx = x_new,fx_new
    return rows, x_new, fx_new, err, k

def fixed_point_iteration(g,x0,tol,max_iter):
    rows=[]
    x=x0
    for k in range(1,max_iter+1):
        x_new = g(x)
        err = abs(x_new - x)
        rows.append({'k':k,'x':x,'g_x':x_new,'err':err})
        if err < tol:
            return rows, x_new, err, k
        x = x_new
    return rows, x_new, err, k

def modified_secant(f,x0,delta,tol,max_iter):
    rows=[]
    x=x0
    for k in range(1,max_iter+1):
        denom = f(x + delta*x) - f(x)
        if denom == 0:
            raise ZeroDivisionError("Denominator zero in modified secant")
        x_new = x - (delta*x)*f(x)/denom
        fnew = f(x_new)
        err = abs(x_new - x)
        rows.append({'k':k,'x':x,'x_new':x_new,'fx':f(x),'fxdx':f(x+delta*x),'fnew':fnew,'err':err})
        if abs(fnew) < tol or err < tol:
            return rows, x_new, fnew, err, k
        x = x_new
    return rows, x_new, fnew, err, k

@app.route("/", methods=["GET","POST"])
def index():
    result = None
    error = None
    if request.method == "POST":
        method = request.form.get("method")
        func_str = request.form.get("function", "").strip()
        tol = float(request.form.get("tol") or 1e-6)
        max_iter = int(request.form.get("max_iter") or 50)
        try:
            f, expr = make_function(func_str)
        except Exception as e:
            error = f"Error parsing f(x): {e}"
            return render_template("index.html", error=error)
        try:
            if method == "bisection":
                a = float(request.form.get("a"))
                b = float(request.form.get("b"))
                rows, root, froot, err, nit = bisection(f,a,b,tol,max_iter)
                result = {'rows':rows, 'root':root, 'froot':froot, 'err':err, 'nit':nit, 'method':'Bisection'}
            elif method == "regula":
                a = float(request.form.get("a"))
                b = float(request.form.get("b"))
                rows, root, froot, err, nit = regula_falsi(f,a,b,tol,max_iter)
                result = {'rows':rows, 'root':root, 'froot':froot, 'err':err, 'nit':nit, 'method':'Regula Falsi'}
            elif method == "secant":
                x0 = float(request.form.get("x0"))
                x1 = float(request.form.get("x1"))
                rows, root, froot, err, nit = secant(f,x0,x1,tol,max_iter)
                result = {'rows':rows, 'root':root, 'froot':froot, 'err':err, 'nit':nit, 'method':'Secant'}
            elif method == "newton":
                fprime_callable, dexpr = derivative_of(expr)
                x0 = float(request.form.get("x0"))
                rows, root, froot, err, nit = newton_raphson(f, fprime_callable, x0, tol, max_iter)
                result = {'rows':rows, 'root':root, 'froot':froot, 'err':err, 'nit':nit, 'method':'Newton-Raphson'}
            elif method == "fixed":
                g_str = request.form.get("g")
                g, _ = make_function(g_str)
                x0 = float(request.form.get("x0"))
                rows, root, err, nit = fixed_point_iteration(g,x0,tol,max_iter)
                result = {'rows':rows, 'root':root, 'err':err, 'nit':nit, 'method':'Fixed Point'}
            elif method == "modified":
                x0 = float(request.form.get("x0"))
                delta = float(request.form.get("delta") or 1e-3)
                rows, root, froot, err, nit = modified_secant(f,x0,delta,tol,max_iter)
                result = {'rows':rows, 'root':root, 'froot':froot, 'err':err, 'nit':nit, 'method':'Modified Secant'}
            else:
                error = "Unknown method selected."
        except Exception as e:
            error = f"Method failed: {e}"
    return render_template("index.html", result=result, error=error)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
