# Zero of Functions (ZOF) Solver

A comprehensive Python application implementing six numerical methods for finding roots of nonlinear equations, with both CLI and Web GUI interfaces.

## Features

### Implemented Numerical Methods

1. **Bisection Method** - Interval halving technique
2. **Regula Falsi Method** - False position method
3. **Secant Method** - Two-point iteration
4. **Newton-Raphson Method** - Tangent line approach
5. **Fixed Point Iteration** - Iterative substitution
6. **Modified Secant Method** - Finite difference approximation

## Project Structure

```
ZOF/
├── ZOF_CLI.py                 # Command-line interface
├── app.py                     # Flask web application
├── requirements.txt           # Python dependencies
├── ZOF_hosted_webGUI_link.txt # Deployment information
├── templates/
│   └── index.html            # Web interface template
└── static/
    └── style.css             # Stylesheet
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:

```bash
git clone [your-repo-url]
cd ZOF
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Command-Line Interface (CLI)

Run the CLI application:

```bash
python ZOF_CLI.py
```

Follow the interactive prompts to:

1. Select a numerical method
2. Enter the function (e.g., `x**2 - 4`, `sin(x) - x/2`)
3. Provide method-specific parameters
4. View iteration details and results

### Web GUI Application

1. Start the Flask server:

```bash
python app.py
```

2. Open your browser and navigate to:

```
http://localhost:5000
```

3. Use the web interface to:
   - Select a numerical method
   - Enter function and parameters
   - View real-time computation results
   - Analyze iteration details in tabular format

## Function Input Syntax

Use standard Python mathematical syntax:

- Powers: `x**2`, `x**3`
- Trigonometric: `sin(x)`, `cos(x)`, `tan(x)`
- Exponential: `exp(x)`, `log(x)`
- Square root: `sqrt(x)`
- Constants: `pi`, `e`

### Examples

- Polynomial: `x**3 - 2*x - 5`
- Trigonometric: `sin(x) - x/2`
- Exponential: `exp(x) - 3*x`
- Mixed: `x*cos(x) - log(x)`

## Method-Specific Requirements

### Bisection & Regula Falsi

- Requires: Left endpoint (a), Right endpoint (b)
- Condition: f(a) and f(b) must have opposite signs

### Secant Method

- Requires: Two initial guesses (x₀, x₁)

### Newton-Raphson

- Requires: One initial guess (x₀)
- Note: Derivative calculated automatically

### Fixed Point Iteration

- Requires: Function g(x) where x = g(x)
- Example: For x² - 4 = 0, use `sqrt(4)` or `(x + 4/x)/2`

### Modified Secant

- Requires: Initial guess (x₀), Perturbation δ
- Default δ: 0.01

## Deployment

### Option 1: Render.com

1. Add `gunicorn` to requirements.txt:

```bash
echo "gunicorn==21.2.0" >> requirements.txt
```

2. Create `Procfile`:

```
web: gunicorn app:app
```

3. Deploy:
   - Push to GitHub
   - Create new Web Service on Render
   - Connect repository
   - Deploy automatically

### Option 2: PythonAnywhere

1. Upload files to PythonAnywhere
2. Install dependencies in console
3. Configure WSGI file
4. Reload web app

### Option 3: Vercel

1. Install Vercel CLI
2. Run `vercel` in project directory
3. Follow deployment prompts

## Output Information

The application provides:

- **Iteration Number**: Current iteration count
- **Computed Values**: Intermediate calculations
- **Function Evaluations**: f(x) values at each step
- **Error Estimates**: Convergence metrics
- **Final Root**: Computed solution
- **Total Iterations**: Number of steps to convergence

## Example Usage

### CLI Example

```
Select method (1-6): 1
Enter function f(x): x**2 - 4
Enter left endpoint (a): 0
Enter right endpoint (b): 3
Enter tolerance: 0.000001
Enter max iterations: 100

Results:
Root: 2.0000000000
Final Error: 9.536743e-07
Iterations: 20
```

### Web Interface

1. Select "Bisection Method"
2. Enter function: `x**2 - 4`
3. Set a = 0, b = 3
4. Click "Solve"
5. View detailed iteration table and final result

## Error Handling

The application handles:

- Invalid function syntax
- Division by zero
- Method divergence
- Incorrect initial values
- Maximum iteration limits

## Technical Stack

- **Backend**: Python 3, Flask
- **Frontend**: HTML5, CSS3, JavaScript
- **Libraries**: SymPy (symbolic math), NumPy-compatible functions
- **Deployment**: Render, PythonAnywhere, Vercel

## Contributing

This is an academic project. For improvements:

1. Fork the repository
2. Create feature branch
3. Submit pull request

## License

Academic project for educational purposes.

## Author

[Your Name]
[Your Matric Number]

## Acknowledgments

- Numerical Methods course materials
- Python Flask documentation
- SymPy library contributors

## Support

For issues or questions:

- Check function syntax
- Verify initial values satisfy method requirements
- Ensure tolerance and max iterations are reasonable
- Review iteration table for convergence patterns
