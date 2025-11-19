#!/usr/bin/env python3
"""
Test script to demonstrate ZOF Solver functionality
Run automated tests of all six methods
"""

from ZOF_CLI import ZOFSolver


def test_all_methods():
    """Test all numerical methods with example problems"""
    solver = ZOFSolver()
    
    print("\n" + "="*70)
    print(" "*20 + "ZOF SOLVER TEST SUITE")
    print("="*70)
    
    # Test 1: Bisection Method
    print("\n1. Testing Bisection Method")
    print("   Function: x**2 - 4")
    print("   Interval: [0, 3]")
    result = solver.bisection_method("x**2 - 4", 0, 3, tol=1e-6, max_iter=100)
    if "error" not in result:
        print(f"   ✓ Root found: {result['root']}")
        print(f"   ✓ Iterations: {result['iterations']}")
        print(f"   ✓ Error: {result['final_error']}")
    else:
        print(f"   ✗ Error: {result['error']}")
    
    # Test 2: Regula Falsi Method
    print("\n2. Testing Regula Falsi Method")
    print("   Function: x**3 - 2*x - 5")
    print("   Interval: [2, 3]")
    result = solver.regula_falsi_method("x**3 - 2*x - 5", 2, 3, tol=1e-6, max_iter=100)
    if "error" not in result:
        print(f"   ✓ Root found: {result['root']}")
        print(f"   ✓ Iterations: {result['iterations']}")
        print(f"   ✓ Error: {result['final_error']}")
    else:
        print(f"   ✗ Error: {result['error']}")
    
    # Test 3: Secant Method
    print("\n3. Testing Secant Method")
    print("   Function: cos(x) - x")
    print("   Initial guesses: x0=0, x1=1")
    result = solver.secant_method("cos(x) - x", 0, 1, tol=1e-6, max_iter=100)
    if "error" not in result:
        print(f"   ✓ Root found: {result['root']}")
        print(f"   ✓ Iterations: {result['iterations']}")
        print(f"   ✓ Error: {result['final_error']}")
    else:
        print(f"   ✗ Error: {result['error']}")
    
    # Test 4: Newton-Raphson Method
    print("\n4. Testing Newton-Raphson Method")
    print("   Function: x**2 - 4")
    print("   Initial guess: x0=3")
    result = solver.newton_raphson_method("x**2 - 4", 3, tol=1e-6, max_iter=100)
    if "error" not in result:
        print(f"   ✓ Root found: {result['root']}")
        print(f"   ✓ Iterations: {result['iterations']}")
        print(f"   ✓ Error: {result['final_error']}")
    else:
        print(f"   ✗ Error: {result['error']}")
    
    # Test 5: Fixed Point Iteration
    print("\n5. Testing Fixed Point Iteration")
    print("   Function g(x): sqrt(x + 2)")
    print("   (Solving x = sqrt(x + 2), i.e., x**2 - x - 2 = 0)")
    print("   Initial guess: x0=1")
    result = solver.fixed_point_iteration("sqrt(x + 2)", 1, tol=1e-6, max_iter=100)
    if "error" not in result:
        print(f"   ✓ Root found: {result['root']}")
        print(f"   ✓ Iterations: {result['iterations']}")
        print(f"   ✓ Error: {result['final_error']}")
    else:
        print(f"   ✗ Error: {result['error']}")
    
    # Test 6: Modified Secant Method
    print("\n6. Testing Modified Secant Method")
    print("   Function: x**2 - 4")
    print("   Initial guess: x0=3, δ=0.01")
    result = solver.modified_secant_method("x**2 - 4", 3, delta=0.01, tol=1e-6, max_iter=100)
    if "error" not in result:
        print(f"   ✓ Root found: {result['root']}")
        print(f"   ✓ Iterations: {result['iterations']}")
        print(f"   ✓ Error: {result['final_error']}")
    else:
        print(f"   ✗ Error: {result['error']}")
    
    print("\n" + "="*70)
    print("All tests completed!")
    print("="*70 + "\n")


def interactive_examples():
    """Provide interactive examples for users"""
    print("\n" + "="*70)
    print(" "*20 + "INTERACTIVE EXAMPLES")
    print("="*70)
    
    examples = [
        {
            "name": "Quadratic Equation",
            "function": "x**2 - 4",
            "description": "Find roots of x² - 4 = 0",
            "method": "Bisection",
            "params": "a=0, b=3"
        },
        {
            "name": "Cubic Equation",
            "function": "x**3 - 2*x - 5",
            "description": "Find roots of x³ - 2x - 5 = 0",
            "method": "Newton-Raphson",
            "params": "x0=2"
        },
        {
            "name": "Transcendental Equation",
            "function": "cos(x) - x",
            "description": "Find roots of cos(x) = x",
            "method": "Secant",
            "params": "x0=0, x1=1"
        },
        {
            "name": "Exponential Equation",
            "function": "exp(x) - 3*x",
            "description": "Find roots of eˣ - 3x = 0",
            "method": "Regula Falsi",
            "params": "a=0, b=2"
        }
    ]
    
    print("\nCommon Root-Finding Examples:\n")
    for i, ex in enumerate(examples, 1):
        print(f"{i}. {ex['name']}")
        print(f"   Function: {ex['function']}")
        print(f"   Description: {ex['description']}")
        print(f"   Suggested Method: {ex['method']}")
        print(f"   Parameters: {ex['params']}")
        print()
    
    print("="*70 + "\n")


if __name__ == "__main__":
    print("\nZOF Solver Test & Examples")
    print("Choose an option:")
    print("1. Run automated tests")
    print("2. View interactive examples")
    print("3. Exit")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        test_all_methods()
    elif choice == "2":
        interactive_examples()
    elif choice == "3":
        print("Goodbye!")
    else:
        print("Invalid choice!")
