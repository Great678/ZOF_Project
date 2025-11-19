#!/usr/bin/env python3
"""
Project Verification Script
Checks if all required files exist and validates basic functionality
"""

import os
import sys

def check_file_exists(filepath, required=True):
    """Check if a file exists"""
    exists = os.path.exists(filepath)
    status = "✓" if exists else "✗"
    req_text = "(REQUIRED)" if required else "(Optional)"
    print(f"  {status} {filepath} {req_text}")
    return exists

def verify_project_structure():
    """Verify all required files exist"""
    print("\n" + "="*70)
    print(" "*20 + "PROJECT STRUCTURE VERIFICATION")
    print("="*70 + "\n")
    
    base_path = "/Users/HP/Downloads/ZOF"
    os.chdir(base_path)
    
    required_files = {
        "ZOF_CLI.py": True,
        "app.py": True,
        "requirements.txt": True,
        "ZOF_hosted_webGUI_link.txt": True,
        "templates/index.html": True,
        "static/style.css": True,
        "README.md": False,
        "Procfile": False,
        ".gitignore": False,
        "DEPLOYMENT_GUIDE.md": False,
        "PROJECT_SUMMARY.md": False,
        "test_solver.py": False,
        "start.sh": False
    }
    
    print("Checking files:")
    missing_required = []
    
    for file, required in required_files.items():
        if not check_file_exists(file, required):
            if required:
                missing_required.append(file)
    
    print()
    
    if missing_required:
        print("❌ Missing required files:")
        for file in missing_required:
            print(f"   - {file}")
        return False
    else:
        print("✅ All required files present!")
        return True

def verify_dependencies():
    """Verify required Python packages can be imported"""
    print("\n" + "="*70)
    print(" "*20 + "DEPENDENCY VERIFICATION")
    print("="*70 + "\n")
    
    dependencies = {
        "flask": "Flask",
        "sympy": "SymPy",
        "werkzeug": "Werkzeug"
    }
    
    missing = []
    
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"  ✓ {name} installed")
        except ImportError:
            print(f"  ✗ {name} NOT installed")
            missing.append(name)
    
    print()
    
    if missing:
        print("❌ Missing dependencies:")
        for dep in missing:
            print(f"   - {dep}")
        print("\n💡 Install with: pip install -r requirements.txt")
        return False
    else:
        print("✅ All dependencies installed!")
        return True

def verify_code_syntax():
    """Verify Python files have valid syntax"""
    print("\n" + "="*70)
    print(" "*20 + "CODE SYNTAX VERIFICATION")
    print("="*70 + "\n")
    
    python_files = ["ZOF_CLI.py", "app.py", "test_solver.py"]
    
    all_valid = True
    
    for file in python_files:
        try:
            with open(file, 'r') as f:
                compile(f.read(), file, 'exec')
            print(f"  ✓ {file} - Valid syntax")
        except SyntaxError as e:
            print(f"  ✗ {file} - Syntax error: {e}")
            all_valid = False
        except FileNotFoundError:
            print(f"  ✗ {file} - File not found")
            all_valid = False
    
    print()
    
    if all_valid:
        print("✅ All Python files have valid syntax!")
        return True
    else:
        print("❌ Some files have syntax errors!")
        return False

def verify_solver_functionality():
    """Test basic solver functionality"""
    print("\n" + "="*70)
    print(" "*20 + "SOLVER FUNCTIONALITY TEST")
    print("="*70 + "\n")
    
    try:
        # Import the solver
        sys.path.insert(0, os.getcwd())
        from ZOF_CLI import ZOFSolver
        
        solver = ZOFSolver()
        
        # Test bisection method with simple equation
        print("  Testing Bisection Method with x**2 - 4...")
        result = solver.bisection_method("x**2 - 4", 0, 3, tol=1e-6, max_iter=100)
        
        if "error" in result:
            print(f"  ✗ Error: {result['error']}")
            return False
        
        root = result['root']
        expected = 2.0
        error = abs(root - expected)
        
        if error < 0.01:
            print(f"  ✓ Found root: {root} (Expected: {expected})")
            print(f"  ✓ Converged in {result['iterations']} iterations")
            print("✅ Solver functionality working!")
            return True
        else:
            print(f"  ✗ Root {root} not close to expected {expected}")
            return False
            
    except Exception as e:
        print(f"  ✗ Error testing solver: {e}")
        return False

def main():
    """Run all verification checks"""
    print("\n" + "="*70)
    print(" "*15 + "ZOF SOLVER - PROJECT VERIFICATION")
    print("="*70)
    
    checks = [
        ("Project Structure", verify_project_structure),
        ("Dependencies", verify_dependencies),
        ("Code Syntax", verify_code_syntax),
        ("Solver Functionality", verify_solver_functionality)
    ]
    
    results = {}
    
    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
        except Exception as e:
            print(f"\n❌ Error running {check_name} check: {e}")
            results[check_name] = False
    
    # Summary
    print("\n" + "="*70)
    print(" "*25 + "VERIFICATION SUMMARY")
    print("="*70 + "\n")
    
    all_passed = True
    for check_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} - {check_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*70)
    
    if all_passed:
        print("\n🎉 ALL CHECKS PASSED! Your project is ready!")
        print("\nNext steps:")
        print("  1. Test CLI: python3 ZOF_CLI.py")
        print("  2. Test Web: python3 app.py (then visit http://localhost:5000)")
        print("  3. Create GitHub repository")
        print("  4. Deploy to Render.com or PythonAnywhere")
        print("  5. Update ZOF_hosted_webGUI_link.txt")
        print("  6. Submit to Scorac.com")
    else:
        print("\n⚠️  Some checks failed. Please review the errors above.")
        print("\nIf dependencies are missing, run:")
        print("  pip install -r requirements.txt")
    
    print("\n" + "="*70 + "\n")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
