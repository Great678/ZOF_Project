"""
Zero of Functions (ZOF) Solver - Web Application
Flask-based web interface for numerical root-finding methods
"""

from flask import Flask, render_template, request, jsonify
import math
import sympy as sp
from typing import Dict

app = Flask(__name__)


class ZOFSolver:
    """Class implementing six root-finding numerical methods"""
    
    def __init__(self):
        self.x = sp.Symbol('x')
        self.iteration_data = []
    
    def parse_function(self, func_str: str):
        """Parse string function to callable"""
        try:
            expr = sp.sympify(func_str)
            func = sp.lambdify(self.x, expr, 'math')
            return func, expr
        except Exception as e:
            raise ValueError(f"Error parsing function: {e}")
    
    def bisection_method(self, func_str: str, a: float, b: float, 
                        tol: float = 1e-6, max_iter: int = 100) -> Dict:
        """Bisection Method"""
        func, expr = self.parse_function(func_str)
        self.iteration_data = []
        
        fa = func(a)
        fb = func(b)
        
        if fa * fb > 0:
            return {"error": "Function must have opposite signs at endpoints"}
        
        for i in range(1, max_iter + 1):
            c = (a + b) / 2
            fc = func(c)
            error = abs(b - a) / 2
            
            self.iteration_data.append({
                "iteration": i,
                "a": round(a, 10),
                "b": round(b, 10),
                "c": round(c, 10),
                "f(c)": round(fc, 10),
                "error": f"{error:.6e}"
            })
            
            if abs(fc) < tol or error < tol:
                return {
                    "root": round(c, 10),
                    "iterations": i,
                    "final_error": f"{error:.6e}",
                    "iteration_data": self.iteration_data,
                    "method": "Bisection Method"
                }
            
            if fa * fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc
        
        return {
            "root": round(c, 10),
            "iterations": max_iter,
            "final_error": f"{error:.6e}",
            "iteration_data": self.iteration_data,
            "method": "Bisection Method",
            "warning": "Maximum iterations reached"
        }
    
    def regula_falsi_method(self, func_str: str, a: float, b: float, 
                           tol: float = 1e-6, max_iter: int = 100) -> Dict:
        """Regula Falsi Method"""
        func, expr = self.parse_function(func_str)
        self.iteration_data = []
        
        fa = func(a)
        fb = func(b)
        
        if fa * fb > 0:
            return {"error": "Function must have opposite signs at endpoints"}
        
        c_old = a
        
        for i in range(1, max_iter + 1):
            c = b - (fb * (b - a)) / (fb - fa)
            fc = func(c)
            
            if i > 1:
                error = abs(c - c_old)
            else:
                error = abs(b - a)
            
            self.iteration_data.append({
                "iteration": i,
                "a": round(a, 10),
                "b": round(b, 10),
                "c": round(c, 10),
                "f(c)": round(fc, 10),
                "error": f"{error:.6e}"
            })
            
            if abs(fc) < tol or error < tol:
                return {
                    "root": round(c, 10),
                    "iterations": i,
                    "final_error": f"{error:.6e}",
                    "iteration_data": self.iteration_data,
                    "method": "Regula Falsi Method"
                }
            
            if fa * fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc
            
            c_old = c
        
        return {
            "root": round(c, 10),
            "iterations": max_iter,
            "final_error": f"{error:.6e}",
            "iteration_data": self.iteration_data,
            "method": "Regula Falsi Method",
            "warning": "Maximum iterations reached"
        }
    
    def secant_method(self, func_str: str, x0: float, x1: float, 
                     tol: float = 1e-6, max_iter: int = 100) -> Dict:
        """Secant Method"""
        func, expr = self.parse_function(func_str)
        self.iteration_data = []
        
        f0 = func(x0)
        f1 = func(x1)
        
        for i in range(1, max_iter + 1):
            if abs(f1 - f0) < 1e-12:
                return {"error": "Division by zero: f(x1) - f(x0) too small"}
            
            x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
            f2 = func(x2)
            error = abs(x2 - x1)
            
            self.iteration_data.append({
                "iteration": i,
                "x0": round(x0, 10),
                "x1": round(x1, 10),
                "x2": round(x2, 10),
                "f(x2)": round(f2, 10),
                "error": f"{error:.6e}"
            })
            
            if abs(f2) < tol or error < tol:
                return {
                    "root": round(x2, 10),
                    "iterations": i,
                    "final_error": f"{error:.6e}",
                    "iteration_data": self.iteration_data,
                    "method": "Secant Method"
                }
            
            x0, f0 = x1, f1
            x1, f1 = x2, f2
        
        return {
            "root": round(x2, 10),
            "iterations": max_iter,
            "final_error": f"{error:.6e}",
            "iteration_data": self.iteration_data,
            "method": "Secant Method",
            "warning": "Maximum iterations reached"
        }
    
    def newton_raphson_method(self, func_str: str, x0: float, 
                             tol: float = 1e-6, max_iter: int = 100) -> Dict:
        """Newton-Raphson Method"""
        func, expr = self.parse_function(func_str)
        self.iteration_data = []
        
        derivative_expr = sp.diff(expr, self.x)
        derivative = sp.lambdify(self.x, derivative_expr, 'math')
        
        x = x0
        
        for i in range(1, max_iter + 1):
            fx = func(x)
            dfx = derivative(x)
            
            if abs(dfx) < 1e-12:
                return {"error": "Derivative too small, division by zero"}
            
            x_new = x - fx / dfx
            error = abs(x_new - x)
            
            self.iteration_data.append({
                "iteration": i,
                "x": round(x, 10),
                "f(x)": round(fx, 10),
                "f'(x)": round(dfx, 10),
                "x_new": round(x_new, 10),
                "error": f"{error:.6e}"
            })
            
            if abs(func(x_new)) < tol or error < tol:
                return {
                    "root": round(x_new, 10),
                    "iterations": i,
                    "final_error": f"{error:.6e}",
                    "iteration_data": self.iteration_data,
                    "method": "Newton-Raphson Method"
                }
            
            x = x_new
        
        return {
            "root": round(x, 10),
            "iterations": max_iter,
            "final_error": f"{error:.6e}",
            "iteration_data": self.iteration_data,
            "method": "Newton-Raphson Method",
            "warning": "Maximum iterations reached"
        }
    
    def fixed_point_iteration(self, func_str: str, x0: float, 
                             tol: float = 1e-6, max_iter: int = 100) -> Dict:
        """Fixed Point Iteration"""
        func, expr = self.parse_function(func_str)
        self.iteration_data = []
        
        x = x0
        
        for i in range(1, max_iter + 1):
            x_new = func(x)
            error = abs(x_new - x)
            
            self.iteration_data.append({
                "iteration": i,
                "x": round(x, 10),
                "g(x)": round(x_new, 10),
                "error": f"{error:.6e}"
            })
            
            if error < tol:
                return {
                    "root": round(x_new, 10),
                    "iterations": i,
                    "final_error": f"{error:.6e}",
                    "iteration_data": self.iteration_data,
                    "method": "Fixed Point Iteration"
                }
            
            if abs(x_new) > 1e10:
                return {"error": "Method diverging, |x| > 10^10"}
            
            x = x_new
        
        return {
            "root": round(x, 10),
            "iterations": max_iter,
            "final_error": f"{error:.6e}",
            "iteration_data": self.iteration_data,
            "method": "Fixed Point Iteration",
            "warning": "Maximum iterations reached"
        }
    
    def modified_secant_method(self, func_str: str, x0: float, delta: float = 0.01, 
                               tol: float = 1e-6, max_iter: int = 100) -> Dict:
        """Modified Secant Method"""
        func, expr = self.parse_function(func_str)
        self.iteration_data = []
        
        x = x0
        
        for i in range(1, max_iter + 1):
            fx = func(x)
            fx_delta = func(x + delta * x) if x != 0 else func(x + delta)
            
            denominator = fx_delta - fx
            if abs(denominator) < 1e-12:
                return {"error": "Division by zero: f(x + δx) - f(x) too small"}
            
            x_new = x - fx * (delta * x if x != 0 else delta) / denominator
            error = abs(x_new - x)
            
            self.iteration_data.append({
                "iteration": i,
                "x": round(x, 10),
                "f(x)": round(fx, 10),
                "f(x+δx)": round(fx_delta, 10),
                "x_new": round(x_new, 10),
                "error": f"{error:.6e}"
            })
            
            if abs(func(x_new)) < tol or error < tol:
                return {
                    "root": round(x_new, 10),
                    "iterations": i,
                    "final_error": f"{error:.6e}",
                    "iteration_data": self.iteration_data,
                    "method": "Modified Secant Method"
                }
            
            x = x_new
        
        return {
            "root": round(x, 10),
            "iterations": max_iter,
            "final_error": f"{error:.6e}",
            "iteration_data": self.iteration_data,
            "method": "Modified Secant Method",
            "warning": "Maximum iterations reached"
        }


# Initialize solver
solver = ZOFSolver()


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/solve', methods=['POST'])
def solve():
    """Handle solve requests"""
    try:
        data = request.json
        method = data.get('method')
        func_str = data.get('function')
        tol = float(data.get('tolerance', 1e-6))
        max_iter = int(data.get('max_iterations', 100))
        
        result = {}
        
        if method == 'bisection':
            a = float(data.get('a'))
            b = float(data.get('b'))
            result = solver.bisection_method(func_str, a, b, tol, max_iter)
        
        elif method == 'regula_falsi':
            a = float(data.get('a'))
            b = float(data.get('b'))
            result = solver.regula_falsi_method(func_str, a, b, tol, max_iter)
        
        elif method == 'secant':
            x0 = float(data.get('x0'))
            x1 = float(data.get('x1'))
            result = solver.secant_method(func_str, x0, x1, tol, max_iter)
        
        elif method == 'newton_raphson':
            x0 = float(data.get('x0'))
            result = solver.newton_raphson_method(func_str, x0, tol, max_iter)
        
        elif method == 'fixed_point':
            x0 = float(data.get('x0'))
            result = solver.fixed_point_iteration(func_str, x0, tol, max_iter)
        
        elif method == 'modified_secant':
            x0 = float(data.get('x0'))
            delta = float(data.get('delta', 0.01))
            result = solver.modified_secant_method(func_str, x0, delta, tol, max_iter)
        
        else:
            result = {"error": "Invalid method selected"}
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
