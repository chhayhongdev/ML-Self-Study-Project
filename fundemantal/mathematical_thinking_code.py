"""
Mathematical Thinking: Practical Exercises and Activities
=========================================================

This module provides practical exercises and activities to develop mathematical
thinking skills - the mental habits and problem-solving approaches that make
you a better mathematician and machine learning practitioner.
"""

import numpy as np
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class PatternRecognitionExercises:
    """Exercises for developing pattern recognition skills"""

    @staticmethod
    def sequence_pattern_analysis():
        """Analyze sequences to find underlying patterns"""
        print("=== Sequence Pattern Analysis ===")

        def check_arithmetic_progression(seq: List[float]) -> Dict[str, Any]:
            """Check if sequence is arithmetic progression"""
            if len(seq) < 3:
                return {}
            diffs = [seq[i+1] - seq[i] for i in range(len(seq)-1)]
            if len(set(diffs)) == 1:
                return {
                    "type": "arithmetic",
                    "common_difference": diffs[0],
                    "next_term": seq[-1] + diffs[0]
                }
            return {}

        def check_geometric_progression(seq: List[float]) -> Dict[str, Any]:
            """Check if sequence is geometric progression"""
            if len(seq) < 3 or any(x == 0 for x in seq):
                return {}
            ratios = [seq[i+1] / seq[i] for i in range(len(seq)-1)]
            if len({round(r, 6) for r in ratios}) == 1:
                return {
                    "type": "geometric",
                    "common_ratio": ratios[0],
                    "next_term": seq[-1] * ratios[0]
                }
            return {}

        def check_quadratic_pattern(seq: List[float]) -> Dict[str, Any]:
            """Check if sequence follows quadratic pattern"""
            if len(seq) < 4:
                return {}
            x = np.arange(len(seq))
            y = np.array(seq)
            x_quad = np.column_stack([x**2, x, np.ones(len(x))])
            try:
                coeffs_quad = np.linalg.lstsq(x_quad, y, rcond=None)[0]
                y_pred_quad = x_quad @ coeffs_quad
                mse_quad = np.mean((y - y_pred_quad)**2)
                if mse_quad < 1e-6:
                    return {
                        "type": "quadratic",
                        "coefficients": coeffs_quad,
                        "equation": f"{coeffs_quad[0]:.2f}x² + {coeffs_quad[1]:.2f}x + "
                                    f"{coeffs_quad[2]:.2f}",
                        "next_term": coeffs_quad[0] * len(seq)**2 + coeffs_quad[1] * len(seq) +
                                    coeffs_quad[2]
                    }
            except np.linalg.LinAlgError:
                pass
            return {}

        def check_special_sequences(seq: List[float]) -> List[Dict[str, Any]]:
            """Check for special known sequences"""
            patterns = []
            if seq == [1, 1, 2, 3, 5, 8, 13]:
                patterns.append({"type": "fibonacci"})
            elif seq == [1, 3, 6, 10, 15, 21]:
                patterns.append({"type": "triangular_numbers"})
            elif all(x**0.5 == int(x**0.5) for x in seq):
                patterns.append({"type": "perfect_squares"})
            return patterns

        def analyze_sequence(seq: List[float]) -> Dict[str, Any]:
            """Comprehensive sequence analysis"""
            result = {"sequence": seq, "patterns": []}

            # Check different pattern types
            result["patterns"].append(check_arithmetic_progression(seq))
            result["patterns"].append(check_geometric_progression(seq))
            result["patterns"].append(check_quadratic_pattern(seq))
            result["patterns"].extend(check_special_sequences(seq))

            # Remove empty patterns
            result["patterns"] = [p for p in result["patterns"] if p]
            return result

        # Test sequences
        test_sequences = [
            [1, 2, 3, 4, 5, 6],
            [2, 4, 8, 16, 32],
            [1, 4, 9, 16, 25],
            [1, 1, 2, 3, 5, 8],
            [1, 3, 6, 10, 15],
            [1, 8, 27, 64, 125],
            [2, 3, 5, 7, 11, 13]
        ]

        for seq in test_sequences:
            analysis = analyze_sequence(seq)
            print(f"\nSequence: {seq}")
            if analysis["patterns"]:
                for pattern in analysis["patterns"]:
                    print(f"  Pattern: {pattern['type']}")
                    if "next_term" in pattern:
                        print(f"  Next term: {pattern['next_term']:.2f}")
                    if "equation" in pattern:
                        print(f"  Equation: {pattern['equation']}")
            else:
                print("  No obvious pattern detected")

    @staticmethod
    def visual_pattern_recognition():
        """Exercises in recognizing visual patterns"""
        print("\n=== Visual Pattern Recognition ===")

        def matrix_completion_puzzle():
            """Complete matrix patterns"""
            print("Matrix Completion Puzzle:")
            print("Complete this pattern:")
            print("2  4  8  ?")
            print("3  6  12 ?")
            print("4  8  16 ?")
            print()
            print("Pattern: Each row multiplies by 2")
            print("Solution: 16, 24, 32")

            # More complex pattern
            print("\nComplex Pattern:")
            print("1  2  4")
            print("2  4  8")
            print("?  8  16")
            print()
            print("Pattern: Each element is 2^(row+col-1)")
            print("Solution: 4 (since 2^(1+2-1) = 2^2 = 4)")

        def shape_sequence_completion():
            """Complete shape sequences"""
            print("\nShape Sequence Completion:")
            print("Circle → Square → Triangle → ?")
            print("Pattern: Increasing number of sides")
            print("Solution: Pentagon (5 sides)")

            print("\nNumber of sides: 0 → 4 → 3 → ?")
            print("Wait, that doesn't make sense...")
            print("Think: Circle has infinite sides, square has 4, triangle has 3")
            print("Pattern: Decreasing number of sides: ∞ → 4 → 3 → 2")
            print("Solution: Line segment (2 endpoints)")

        matrix_completion_puzzle()
        shape_sequence_completion()

    @staticmethod
    def data_pattern_exploration():
        """Explore patterns in real datasets"""
        print("\n=== Data Pattern Exploration ===")

        # Generate synthetic datasets with known patterns
        np.random.seed(42)
        rng = np.random.default_rng(42)

        # Linear relationship
        x_linear = np.linspace(0, 10, 50)
        y_linear = 2 * x_linear + 1 + rng.normal(0, 1, 50)

        # Quadratic relationship
        x_quad = np.linspace(-5, 5, 50)
        y_quad = x_quad**2 + rng.normal(0, 2, 50)

        # Periodic pattern
        x_periodic = np.linspace(0, 4*np.pi, 100)
        y_periodic = np.sin(x_periodic) + 0.5*np.cos(2*x_periodic) + rng.normal(0, 0.3, 100)

        datasets = [
            ("Linear", x_linear, y_linear),
            ("Quadratic", x_quad, y_quad),
            ("Periodic", x_periodic, y_periodic)
        ]

        for name, x, y in datasets:
            print(f"\n{name} Dataset Analysis:")

            # Simple statistical measures
            correlation = np.corrcoef(x, y)[0, 1]
            print(f"  Correlation coefficient: {correlation:.3f}")

            # Try linear fit
            coeffs_linear = np.polyfit(x, y, 1)
            y_pred_linear = np.polyval(coeffs_linear, x)
            mse_linear = np.mean((y - y_pred_linear)**2)

            # Try quadratic fit
            coeffs_quad = np.polyfit(x, y, 2)
            y_pred_quad = np.polyval(coeffs_quad, x)
            mse_quad = np.mean((y - y_pred_quad)**2)

            print(f"  Linear fit MSE: {mse_linear:.3f}")
            print(f"  Quadratic fit MSE: {mse_quad:.3f}")

            if mse_quad < mse_linear * 0.7:
                print("  Quadratic pattern detected!")
            elif abs(correlation) > 0.8:
                print("  Strong linear pattern detected!")
            else:
                print("  Pattern requires further analysis")


class AbstractionExercises:
    """Exercises for developing abstraction skills"""

    DIVIDE_CONQUER = "Divide and conquer"

    @staticmethod
    def function_generalization():
        """Practice generalizing from specific functions"""
        print("=== Function Generalization Exercises ===")

        def generalize_operations():
            """Generalize specific operations to abstract concepts"""

            print("Specific Operations → Abstract Concepts:")

            examples = [
                (["add two numbers", "add three numbers", "add n numbers"],
                 "Summation (Σ) - associative and commutative operation"),

                (["multiply by 2", "multiply by 3", "multiply by n"],
                 "Scaling transformation - linear operator"),

                (["sort list", "find maximum", "find minimum"],
                 "Order statistics - comparison-based algorithms"),

                (["count items", "measure length", "calculate area"],
                 "Measure theory - assigning values to sets")
            ]

            for specific, abstract in examples:
                print(f"\nSpecific cases: {', '.join(specific)}")
                print(f"Abstract generalization: {abstract}")

        def functional_composition():
            """Understand function composition"""
            print("\nFunction Composition:")

            # Define some basic functions
            def add_one(x): return x + 1
            def multiply_two(x): return x * 2
            def square(x): return x ** 2

            functions = [add_one, multiply_two, square]
            names = ["+1", "×2", "²"]

            # Test different compositions
            x = 3
            print(f"Starting with x = {x}")

            def compose_functions(f, g, x_val):
                return f(g(x_val))

            for i in range(len(functions)):
                for j in range(len(functions)):
                    if i != j:  # Avoid self-composition for clarity
                        result = compose_functions(functions[i], functions[j], x)
                        print(f"({names[j]} ∘ {names[i]})({x}) = "
                              f"{names[j]}({names[i]}({x})) = {result}")

        generalize_operations()
        functional_composition()

    @staticmethod
    def algorithm_abstraction():
        """Abstract algorithms to general principles"""
        print("\n=== Algorithm Abstraction ===")

        def sorting_algorithm_family():
            """Understand the sorting algorithm family"""
            print("Sorting Algorithms - Abstract Framework:")

            sorting_concepts = {
                "Comparison-based": "Compare elements pairwise",
                AbstractionExercises.DIVIDE_CONQUER: "Split, sort subproblems, merge",
                "In-place": "Sort without extra space",
                "Stable": "Preserve relative order of equal elements",
                "Adaptive": "Perform better on partially sorted data"
            }

            for concept, description in sorting_concepts.items():
                print(f"  {concept}: {description}")

            # Concrete examples
            examples = [
                ("Bubble sort", ["Comparison-based", "In-place", "Stable"]),
                ("Quick sort", ["Comparison-based", AbstractionExercises.DIVIDE_CONQUER,
                                "In-place"]),
                ("Merge sort", ["Comparison-based",
                                AbstractionExercises.DIVIDE_CONQUER, "Stable"]),
                ("Heap sort", ["Comparison-based", "In-place"])
            ]

            print("\nAlgorithm Classification:")
            for algo, properties in examples:
                print(f"  {algo}: {', '.join(properties)}")

        def search_algorithm_abstraction():
            """Abstract search algorithms"""
            print("\nSearch Algorithms - Abstract Framework:")

            search_concepts = {
                "State space": "Set of all possible configurations",
                "Goal test": "Function determining if state is solution",
                "Successors": "Function generating next states",
                "Cost function": "Measure of path quality",
                "Heuristic": "Estimate of distance to goal"
            }

            for concept, description in search_concepts.items():
                print(f"  {concept}: {description}")

        sorting_algorithm_family()
        search_algorithm_abstraction()

    @staticmethod
    def ml_model_abstraction():
        """Abstract machine learning models"""
        print("\n=== ML Model Abstraction ===")

        def model_family_analysis():
            """Analyze model families abstractly"""

            model_families = {
                "Linear Models": {
                    "hypothesis_class": "Linear combinations of features",
                    "capacity": "Limited by feature engineering",
                    "bias_variance": "High bias, low variance",
                    "interpretability": "High"
                },

                "Tree-based Models": {
                    "hypothesis_class": "Piecewise constant functions",
                    "capacity": "High (can overfit)",
                    "bias_variance": "Low bias, high variance",
                    "interpretability": "Medium"
                },

                "Neural Networks": {
                    "hypothesis_class": "Compositions of non-linear functions",
                    "capacity": "Very high (universal approximation)",
                    "bias_variance": "Low bias, very high variance",
                    "interpretability": "Low"
                }
            }

            for family, properties in model_families.items():
                print(f"\n{family}:")
                for prop, desc in properties.items():
                    print(f"  {prop}: {desc}")

        model_family_analysis()


class ProblemSolvingExercises:
    """Exercises using systematic problem-solving approaches"""

    @staticmethod
    def polya_method_practice():
        """Practice Polya's four-step method"""
        print("=== Polya's Method Practice ===")

        def solve_geometry_problem():
            """Apply Polya's method to a geometry problem"""
            print("Problem: Find the area of a circle given radius r")
            print()

            print("1. UNDERSTAND THE PROBLEM:")
            print("   - We have a circle with radius r")
            print("   - We need to find its area")
            print("   - Area should be a function of r")
            print("   - We know area grows with radius")
            print()

            print("2. DEVISE A PLAN:")
            print("   - Approximate with polygons")
            print("   - Take limit as number of sides → ∞")
            print("   - Use integral calculus")
            print("   - Use known formula and verify")
            print()

            print("3. CARRY OUT THE PLAN:")
            print("   Using approximation method:")
            print("   - Square: side 2r, area (2r)² = 4r²")
            print("   - Octagon: area ≈ 4.828r²")
            print("   - 16-gon: area ≈ 6.122r²")
            print("   - 32-gon: area ≈ 6.242r²")
            print("   - As sides → ∞, area → πr² ≈ 3.1416r²")
            print()

            print("4. LOOK BACK:")
            print("   - Does this make sense? Area should be proportional to r²")
            print("   - Check special cases: r=1, area=π≈3.14")
            print("   - Verify with known values")
            print("   - Generalize: area scales with square of linear dimensions")

            # Numerical verification
            r = 2.0
            area = np.pi * r**2
            print(f"   Numerical check: r={r}, area={area:.2f}")

        solve_geometry_problem()

    @staticmethod
    def dimensional_analysis_practice():
        """Practice dimensional analysis"""
        print("\n=== Dimensional Analysis Practice ===")

        def population_density_problem():
            """Estimate population density using dimensional analysis"""
            print("Problem: Estimate the population density of your city")
            print()

            print("Known quantities:")
            print("- City area: ~ 100-1000 km²")
            print("- Population: ~ 100,000 - 10,000,000")
            print("- Building height: ~ 10-50 meters")
            print("- Street width: ~ 10-20 meters")
            print()

            print("Dimensional analysis:")
            print("Density = Population / Area")
            print("But we can estimate using more fundamental quantities:")
            print()

            # Rough estimation
            floors_per_building = 10
            people_per_floor = 20
            building_footprint = 20 * 20  # meters²
            buildings_per_km2 = 50  # rough estimate

            density_estimate = (buildings_per_km2 * floors_per_building *
                                people_per_floor / (building_footprint / 1000000))

            print("Estimation method:")
            print(f"- Buildings per km²: {buildings_per_km2}")
            print(f"- Floors per building: {floors_per_building}")
            print(f"- People per floor: {people_per_floor}")
            print(f"- Building footprint: {building_footprint}m²")
            print(f"- Estimated density: {density_estimate:.0f} people/km²")

            print("\nReality check: Typical city densities range from 1,000-20,000 people/km²")

        population_density_problem()

    @staticmethod
    def case_analysis_method():
        """Practice case analysis for problem solving"""
        print("\n=== Case Analysis Method ===")

        def parity_problem():
            """Solve a problem by considering different cases"""
            print("Problem: Prove that for any integer n, n² + n is always even")
            print()

            print("Case Analysis:")
            print("Case 1: n is even")
            print("  n = 2k for some integer k")
            print("  n² + n = (2k)² + 2k = 4k² + 2k = 2(2k² + k) = 2m")
            print("  Since m = 2k² + k is an integer, result is even")
            print()

            print("Case 2: n is odd")
            print("  n = 2k + 1 for some integer k")
            print("  n² + n = (2k + 1)² + (2k + 1) = 4k² + 4k + 1 + 2k + 1 = "
                  "4k² + 6k + 2 = 2(2k² + 3k + 1) = 2m")
            print("  Since m = 2k² + 3k + 1 is an integer, result is even")
            print()

            print("Since both cases lead to even results, the statement is proved.")

            # Verification
            test_values = list(range(-5, 6))
            results = [(n, n**2 + n, (n**2 + n) % 2 == 0) for n in test_values]

            print("\nVerification:")
            for n, expr, is_even in results:
                print(f"n={n:2d}: {expr:3d} is {'even' if is_even else 'odd'}")

        parity_problem()


class CriticalThinkingExercises:
    """Exercises for developing mathematical critical thinking"""

    @staticmethod
    def assumption_identification():
        """Identify and question assumptions"""
        print("=== Assumption Identification ===")

        def analyze_ml_assumptions():
            """Analyze assumptions in ML statements"""

            statements = [
                "Neural networks can learn any function with enough data",
                "Cross-validation eliminates overfitting",
                "Feature scaling always improves performance",
                "Deep learning models are better than shallow ones"
            ]

            for statement in statements:
                print(f"\nStatement: '{statement}'")
                print("Hidden assumptions:")

                if "neural networks" in statement:
                    print("- The function is in the hypothesis class")
                    print("- Sufficient model capacity (width/depth)")
                    print("- Data is representative of the target distribution")
                    print("- Optimization converges to a good solution")
                    print("- No adversarial examples or distribution shift")

                elif "cross-validation" in statement:
                    print("- Validation set is representative")
                    print("- No data leakage between folds")
                    print("- Model selection criteria are appropriate")
                    print("- Computational resources are sufficient")

                elif "feature scaling" in statement:
                    print("- Features have different scales")
                    print("- Algorithm is sensitive to feature scales")
                    print("- Scaling doesn't change relative relationships")
                    print("- Target variable isn't affected")

                elif "deep learning" in statement:
                    print("- Sufficient training data available")
                    print("- Computational resources available")
                    print("- Problem requires complex representations")
                    print("- Interpretability isn't critical")

        analyze_ml_assumptions()

    @staticmethod
    def counterexample_hunting():
        """Find counterexamples to common intuitions"""
        print("\n=== Counterexample Hunting ===")

        def find_counterexamples():
            """Find counterexamples to ML myths"""

            myths_and_counterexamples = {
                "More parameters always mean better models": [
                    "Linear regression often outperforms complex models on small datasets",
                    "Overparameterized models can memorize training data"
                ],

                "Higher accuracy is always better": [
                    "Overfitted model: 99% training accuracy, 60% test accuracy",
                    "Sometimes precision/recall trade-offs matter more"
                ],

                "Neural networks are uninterpretable": [
                    "Feature importance analysis works for many models",
                    "Simple networks can be manually inspected",
                    "Some architectures have built-in interpretability"
                ],

                "Big data solves everything": [
                    "Biased data leads to biased models regardless of size",
                    "Some problems require domain knowledge, not just data"
                ]
            }

            for myth, counterexamples in myths_and_counterexamples.items():
                print(f"\nMyth: {myth}")
                print("Counterexamples:")
                for ce in counterexamples:
                    print(f"  • {ce}")

        find_counterexamples()

    @staticmethod
    def proof_technique_practice():
        """Practice different proof techniques"""
        print("\n=== Proof Technique Practice ===")

        def direct_proof_example():
            """Practice direct proof"""
            print("Direct Proof: Sum of first n odd numbers is n²")
            print("Proof: The k-th odd number is 2k-1")
            print("Sum = (2*1-1) + (2*2-1) + ... + (2*n-1) = 2(1+2+...+n) - n = "
                  "2(n(n+1)/2) - n = n²")

            # Verification
            for n in range(1, 6):
                odds = [2*k-1 for k in range(1, n+1)]
                sum_odds = sum(odds)
                n_squared = n**2
                print(f"n={n}: odds={odds}, sum={sum_odds}, n²={n_squared}, "
                      f"equal={sum_odds == n_squared}")

        def induction_practice():
            """Practice mathematical induction"""
            print("\nInduction: Prove 1 + 3 + 5 + ... + (2n-1) = n²")

            def odd_sum_formula(n):
                return sum(2*k-1 for k in range(1, n+1))

            print("Base case (n=1): 1 = 1² ✓")

            print("Inductive step: Assume true for n=k, prove for n=k+1")
            print("S(k) = 1 + 3 + ... + (2k-1) = k²")
            print("S(k+1) = S(k) + (2(k+1)-1) = k² + (2k+1) = k² + 2k + 1 = "
                  "(k+1)² ✓")

            # Verification
            for n in range(1, 8):
                formula_result = n**2
                actual_sum = odd_sum_formula(n)
                print(f"n={n}: formula={formula_result}, actual={actual_sum}, "
                      f"match={formula_result == actual_sum}")

        direct_proof_example()
        induction_practice()


class DailyMentalExercises:
    """Daily exercises to build mathematical thinking habits"""

    @staticmethod
    def estimation_challenges():
        """Practice estimation skills"""
        print("=== Daily Estimation Challenges ===")

        challenges = [
            {
                "question": "How many tennis balls fit in a Boeing 747?",
                "hints": ["747 volume ≈ 300m × 6m × 6m", "Tennis ball diameter ≈ 6.7cm"],
                "solution": "~20 million balls"
            },
            {
                "question": "How long would it take to count to 1 million?",
                "hints": ["~2-3 numbers per second", "8 hours/day"],
                "solution": "~5-6 days"
            },
            {
                "question": "What's the weight of all humans on Earth?",
                "hints": ["World population ≈ 8 billion", "Average weight ≈ 60kg"],
                "solution": "~480 billion kg = 480 million tons"
            }
        ]

        for challenge in challenges:
            print(f"\n{challenge['question']}")
            print("Hints:")
            for hint in challenge['hints']:
                print(f"  • {hint}")
            print(f"Solution: {challenge['solution']}")

    @staticmethod
    def pattern_drills():
        """Quick pattern recognition drills"""
        print("\n=== Pattern Recognition Drills ===")

        # Number pattern drill
        def number_pattern_drill():
            patterns = [
                ([1, 4, 7, 10], "Add 3"),
                ([2, 6, 18, 54], "Multiply by 3"),
                ([1, 8, 27, 64], "n³"),
                ([1, 11, 21, 1211], "Look and say sequence")
            ]

            print("Identify the pattern and find next term:")
            for seq, pattern in patterns:
                print(f"{seq} → ? (Pattern: {pattern})")

        # Logic puzzle
        def logic_puzzle():
            print("\nLogic Puzzle:")
            print("If all bloops are razzes and some razzes are fizzles,")
            print("can we conclude that some bloops are fizzles?")
            print("Answer: No - this is the classic fallacy of the undistributed middle")

        number_pattern_drill()
        logic_puzzle()

    @staticmethod
    def abstraction_practice():
        """Practice abstracting concrete problems"""
        print("\n=== Abstraction Practice ===")

        concrete_abstract_pairs = [
            ("Finding the shortest path in a city",
             "Graph search with edge weights"),

            ("Sorting student grades",
             "Comparison-based sorting"),

            ("Predicting house prices",
             "Regression in high-dimensional space"),

            ("Recognizing handwritten digits",
             "Classification with invariance to transformations")
        ]

        for concrete, abstract in concrete_abstract_pairs:
            print(f"Concrete: {concrete}")
            print(f"Abstract: {abstract}")
            print()

    @staticmethod
    def ml_geometry_thinking():
        """Think about ML problems geometrically"""
        print("=== ML Geometry Thinking ===")

        geometric_concepts = {
            "Decision boundary": "Hyperplane separating classes in feature space",
            "Manifold learning": "Data lies on low-dimensional surface in high-D space",
            "Curse of dimensionality": "Volume grows exponentially with dimension",
            "Kernel trick": "Implicitly mapping to higher dimensions"
        }

        for concept, explanation in geometric_concepts.items():
            print(f"{concept}: {explanation}")

        # Practical exercise
        print("\nPractical Exercise:")
        print("Think about how these concepts apply to:")
        print("- Why PCA works for dimensionality reduction")
        print("- Why SVMs use kernels")
        print("- Why deep networks can represent complex functions")


def main():
    """Run all mathematical thinking exercises"""
    print("Mathematical Thinking: Practical Exercises and Activities")
    print("=" * 65)

    # Pattern Recognition
    pattern_exercises = PatternRecognitionExercises()
    pattern_exercises.sequence_pattern_analysis()
    pattern_exercises.visual_pattern_recognition()
    pattern_exercises.data_pattern_exploration()

    # Abstraction
    abstraction_exercises = AbstractionExercises()
    abstraction_exercises.function_generalization()
    abstraction_exercises.algorithm_abstraction()
    abstraction_exercises.ml_model_abstraction()

    # Problem Solving
    problem_exercises = ProblemSolvingExercises()
    problem_exercises.polya_method_practice()
    problem_exercises.dimensional_analysis_practice()
    problem_exercises.case_analysis_method()

    # Critical Thinking
    critical_exercises = CriticalThinkingExercises()
    critical_exercises.assumption_identification()
    critical_exercises.counterexample_hunting()
    critical_exercises.proof_technique_practice()

    # Daily Exercises
    daily_exercises = DailyMentalExercises()
    daily_exercises.estimation_challenges()
    daily_exercises.pattern_drills()
    daily_exercises.abstraction_practice()
    daily_exercises.ml_geometry_thinking()

    print("\n" + "=" * 65)
    print("All mathematical thinking exercises completed!")
    print("\nRemember: Mathematical thinking is a skill that improves with daily practice.")
    print("Try to incorporate these exercises into your daily routine!")


if __name__ == "__main__":
    main()
