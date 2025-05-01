"""
Answer evaluation module that extracts and validates answers from LLM generations.
"""
import re
from typing import Dict, List, Any, Optional, Union, Tuple
import sympy
from sympy.parsing import sympy_parser
from pylatexenc import latex2text

class AnswerEvaluator:
    """
    Extracts answers from model generations and evaluates their correctness.
    """
    
    def __init__(self):
        """Initialize the answer evaluator."""
        pass
    
    def evaluate(self, generation: str, ground_truth: str) -> Tuple[Optional[str], bool]:
        """
        Evaluate a single generation against a ground truth.
        
        Args:
            generation: Model output text
            ground_truth: Ground truth answer
            
        Returns:
            Tuple of (extracted_answer, is_correct)
        """
        extracted_answer = self.extract_answer(generation)
        correctness = self.evaluate_correctness(extracted_answer, ground_truth)
        return extracted_answer, correctness.get("any_method_correct", False)
    
    def evaluate_batch(self, generations: List[str], ground_truths: List[str]) -> List[Tuple[Optional[str], bool]]:
        """
        Evaluate a batch of generations against ground truths.
        
        Args:
            generations: List of model output texts
            ground_truths: List of ground truth answers
            
        Returns:
            List of tuples (extracted_answer, is_correct)
        """
        if len(generations) != len(ground_truths):
            raise ValueError("Length of generations and ground_truths must match")
        
        results = []
        for gen, gt in zip(generations, ground_truths):
            results.append(self.evaluate(gen, gt))
        
        return results
    
    def extract_answer(self, text: str) -> Optional[str]:
        """
        Extract answer from model output text.
        
        Args:
            text: Model output text
            
        Returns:
            Extracted answer or None if not found
        """
        # Try to extract boxed answer first
        boxed_answer = self.extract_boxed_answer(text)
        if boxed_answer:
            return boxed_answer
        
        # Otherwise, try to find answer patterns
        answer_patterns = [
            r"(?:The answer is|Thus, the answer is|Therefore, the answer is|The final answer is|So, the answer is)[:\s]+([^\n\.]+)",
            r"(?:Answer|Result)[:\s]+([^\n\.]+)",
            r"(?:So|Thus|Therefore|Hence),[:\s]+([^\n\.]+)",
        ]
        
        for pattern in answer_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Return the last match
                return matches[-1].strip()
        
        # If no pattern matches, try to find the last line with a number
        lines = text.strip().split('\n')
        for line in reversed(lines):
            line = line.strip()
            if re.search(r'\d', line) and len(line) < 100:
                return line
        
        # If all fails, return None
        return None
    
    def evaluate_correctness(self, extracted_answer: str, ground_truth: str) -> Dict[str, Any]:
        """
        Evaluate the correctness of an extracted answer.
        
        Args:
            extracted_answer: Extracted answer
            ground_truth: Ground truth answer
            
        Returns:
            Dictionary with evaluation results
        """
        if not extracted_answer:
            return {
                "exact_match": False,
                "math_verify": False,
                "mathd_normalize": False,
                "sympy_verify": False,
                "any_method_correct": False,
                "error": "No answer extracted",
            }
        
        try:
            results = self.evaluate_math_answer(extracted_answer, ground_truth)
            return results
        except Exception as e:
            return {
                "exact_match": False,
                "math_verify": False,
                "mathd_normalize": False,
                "sympy_verify": False,
                "any_method_correct": False,
                "error": str(e),
            }
    
    # Original methods for answer extraction and validation
    def extract_boxed_answer(self, text: str) -> Optional[str]:
        """
        Extract answer from text using the specified boxed pattern.
        Takes the last match of the pattern.
        
        Args:
            text: Text to extract answer from
            
        Returns:
            Extracted answer or None if not found
        """
        # Look for \boxed{...} pattern - handles nested braces correctly
        boxed_pattern = re.compile(r"\\boxed\{((?:[^{}]|\\{|\\}|(?:\{(?:[^{}]|\\{|\\}|(?:\{(?:[^{}]|\\{|\\}|(?:\{[^{}]*\}))*\}))*\}))*?)\}")
        matches = boxed_pattern.findall(text)
        
        if matches:
            # Return the last match
            return matches[-1]
        
        return None

    def evaluate_math_answer(self, answer: str, ground_truth: str) -> Dict[str, Any]:
        """
        Evaluates a math answer against ground truth using multiple methods.
        
        Args:
            answer (str): The answer to evaluate (already extracted from boxed pattern)
            ground_truth (str): The correct answer
            
        Returns:
            dict: Dictionary with evaluation results for each method
        """
        results = {}
        
        # Method 1: Simple string equivalence
        results["exact_match"] = answer.strip() == ground_truth.strip()
        
        # Method 2: Math Verify using parse and verify
        try:
            from math_verify import parse, verify
            parsed_answer = parse(self._prepare_for_math_verify(answer))
            parsed_ground_truth = parse(self._prepare_for_math_verify(ground_truth))
            results["math_verify"] = verify(parsed_ground_truth, parsed_answer)
        except Exception as e:
            results["math_verify"] = False
            results["math_verify_error"] = str(e)
        
        # Method 3: MathD normalization
        try:
            # First, normalize both answers using mathd_normalize_answer
            normalized_answer = self._mathd_normalize_answer(answer)
            normalized_ground_truth = self._mathd_normalize_answer(ground_truth)
            
            # Compare the normalized answers
            results["mathd_normalize"] = normalized_answer == normalized_ground_truth
        except Exception as e:
            results["mathd_normalize"] = False
            results["mathd_normalize_error"] = str(e)
        
        # Method 4: SymPy-based verification
        try:
            # Process both answers with the sympy prep function
            sympy_answer = self._sympy_prepare_answer(answer)
            sympy_ground_truth = self._sympy_prepare_answer(ground_truth)
            
            # Now check using the original grade_answer_sympy function
            results["sympy_verify"] = self._grade_answer_sympy(answer, ground_truth)
        except Exception as e:
            results["sympy_verify"] = False
            results["sympy_verify_error"] = str(e)
        
        # Add a summary field for convenience
        results["any_method_correct"] = any([
            results["exact_match"],
            results.get("math_verify", False),
            results.get("mathd_normalize", False),
            results.get("sympy_verify", False)
        ])
        
        return results

    # Helper methods
    def _prepare_for_math_verify(self, answer: str) -> str:
        """
        Prepares an answer for the math_verify module.
        The original code doesn't show specific preprocessing for math_verify,
        so this is a placeholder that does minimal formatting.
        
        Args:
            answer: The answer string to prepare
            
        Returns:
            The prepared answer string
        """
        # The original code doesn't show specific preprocessing for math_verify,
        # so we'll just do minimal formatting
        return answer.strip()

    def _mathd_normalize_answer(self, answer: Optional[str]) -> Optional[str]:
        """
        Normalizes an answer string according to the MathD normalization scheme.
        
        Args:
            answer: The answer string to normalize
            
        Returns:
            The normalized answer string
        """
        if answer is None:
            return None
        answer = answer.strip()
        try:
            # Remove enclosing `\text{}`.
            m = re.search(r"^\\text\{(?P<text>.+?)\}$", answer)
            if m is not None:
                answer = m.group("text").strip()
            return self._strip_string(answer)
        except:
            return answer

    def _strip_string(self, string: str) -> str:
        """Helper function for mathd_normalize_answer to strip and normalize LaTeX formatting."""
        def _fix_a_slash_b(string):
            if len(string.split("/")) != 2:
                return string
            a = string.split("/")[0]
            b = string.split("/")[1]
            try:
                a = int(a)
                b = int(b)
                assert string == "{}/{}".format(a, b)
                new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
                return new_string
            except:
                return string

        def _remove_right_units(string):
            # "\\text{ " only ever occurs (at least in the val set) when describing units
            if "\\text{ " in string:
                splits = string.split("\\text{ ")
                assert len(splits) == 2
                return splits[0]
            else:
                return string

        def _fix_sqrt(string):
            if "\\sqrt" not in string:
                return string
            splits = string.split("\\sqrt")
            new_string = splits[0]
            for split in splits[1:]:
                if split[0] != "{":
                    a = split[0]
                    new_substr = "\\sqrt{" + a + "}" + split[1:]
                else:
                    new_substr = "\\sqrt" + split
                new_string += new_substr
            return new_string
        
        def _fix_fracs(string):
            """
            Fix various fraction representations in LaTeX:
            - \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}
            - Works with \frac1{72} (but not \frac{72}1)
            - Also converts a/b --> \frac{a}{b}
            """
            # Match pattern like \frac1{...} or \frac12
            p1 = re.compile(r"\\frac([0-9]+)([a-zA-Z]+|[0-9]+)")
            while True:
                next_string = p1.sub(r"\\frac{\1}{\2}", string)
                if next_string == string:
                    break
                string = next_string
                
            # Match pattern like \frac1{...}
            p2 = re.compile(r"\\frac([0-9]+)(\{[^\}]+\})")
            while True:
                next_string = p2.sub(r"\\frac{\1}\2", string)
                if next_string == string:
                    break
                string = next_string
                
            # Convert simple a/b to \frac{a}{b}
            p3 = re.compile(r"([0-9]+)/([0-9]+)")
            while True:
                next_string = p3.sub(r"\\frac{\1}{\2}", string)
                if next_string == string:
                    break
                string = next_string
                
            return string
            
        # linebreaks
        string = string.replace("\n", "")

        # remove inverse spaces
        string = string.replace("\\!", "")

        # replace \\ with \
        string = string.replace("\\\\", "\\")

        # replace tfrac and dfrac with frac
        string = string.replace("tfrac", "frac")
        string = string.replace("dfrac", "frac")

        # remove \left and \right
        string = string.replace("\\left", "")
        string = string.replace("\\right", "")

        # Remove circ (degrees)
        string = string.replace("^{\\circ}", "")
        string = string.replace("^\\circ", "")

        # remove dollar signs
        string = string.replace("\\$", "")

        # remove units (on the right)
        string = _remove_right_units(string)

        # remove percentage
        string = string.replace("\\%", "")
        string = string.replace(r"\%", "")

        # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
        string = string.replace(" .", " 0.")
        string = string.replace("{.", "{0.")
        # if empty, return empty string
        if len(string) == 0:
            return string
        if string[0] == ".":
            string = "0" + string

        # to consider: get rid of e.g. "k = " or "q = " at beginning
        if len(string.split("=")) == 2:
            if len(string.split("=")[0]) <= 2:
                string = string.split("=")[1]

        # fix sqrt3 --> sqrt{3}
        string = _fix_sqrt(string)

        # remove spaces
        string = string.replace(" ", "")

        # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
        string = _fix_fracs(string)

        # manually change 0.5 --> \frac{1}{2}
        if string == "0.5":
            string = "\\frac{1}{2}"

        # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
        string = _fix_a_slash_b(string)

        return string

    def _sympy_prepare_answer(self, answer: str) -> str:
        """Prepares an answer for sympy verification by normalizing it."""
        return self._normalize(answer)

    def _normalize(self, expr: str) -> str:
        """Normalize answer expressions for sympy verification."""
        if expr is None:
            return None

        # Remove enclosing `\text{}`.
        m = re.search(r"^\\text\{(?P<text>.+?)\}$", expr)
        if m is not None:
            expr = m.group("text")

        expr = expr.replace("\\%", "%")
        expr = expr.replace("\\$", "$")
        expr = expr.replace("$", "")
        expr = expr.replace("%", "")
        expr = expr.replace(" or ", " , ")
        expr = expr.replace(" and ", " , ")

        expr = expr.replace("million", "*10^6")
        expr = expr.replace("billion", "*10^9")
        expr = expr.replace("trillion", "*10^12")

        for unit in [
            "degree",
            "cm",
            "centimeter",
            "meter",
            "mile",
            "second",
            "minute",
            "hour",
            "day",
            "week",
            "month",
            "year",
            "foot",
            "feet",
            "inch",
            "yard",
        ]:
            expr = re.sub(f"{unit}(es)?(s)? *(\\^[0-9]+)?", "", expr)
        expr = re.sub(f"\\^ *\\\\circ", "", expr)

        if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
            expr = expr[1:-1]

        expr = re.sub(",\\\\! *", "", expr)
        if self._is_float(expr) and self._is_int(float(expr)):
            expr = str(int(round(float(expr))))
        if "\\" in expr:
            try:
                expr = self._parse_latex(expr)
            except:
                pass

        # edge case with mixed numbers and negative signs
        expr = re.sub("- *", "-", expr)

        expr = self._inject_implicit_mixed_number(expr)
        expr = expr.replace(" ", "")

        # if we somehow still have latex braces here, just drop them
        expr = expr.replace("{", "")
        expr = expr.replace("}", "")

        # don't be case sensitive for text answers
        expr = expr.lower()

        if self._str_is_int(expr):
            expr = str(self._str_to_int(expr))

        return expr

    def _parse_latex(self, expr: str) -> str:
        """Attempts to parse latex to an expression sympy can read."""
        expr = expr.replace("\\tfrac", "\\frac")
        expr = expr.replace("\\dfrac", "\\frac")
        expr = expr.replace("\\frac", " \\frac")  # Play nice with mixed numbers.
        expr = latex2text.LatexNodes2Text().latex_to_text(expr)

        # Replace the specific characters that this parser uses.
        expr = expr.replace("√", "sqrt")
        expr = expr.replace("π", "pi")
        expr = expr.replace("∞", "inf")
        expr = expr.replace("∪", "U")
        expr = expr.replace("·", "*")
        expr = expr.replace("×", "*")

        return expr.strip()

    def _is_float(self, num: str) -> bool:
        try:
            float(num)
            return True
        except ValueError:
            return False

    def _is_int(self, x: float) -> bool:
        try:
            return abs(x - int(round(x))) <= 1e-7
        except:
            return False

    def _is_frac(self, expr: str) -> bool:
        return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))

    def _str_is_int(self, x: str) -> bool:
        try:
            x = self._strip_properly_formatted_commas(x)
            x = float(x)
            return abs(x - int(round(x))) <= 1e-7
        except:
            return False

    def _str_to_int(self, x: str) -> int:
        x = x.replace(",", "")
        x = float(x)
        return int(x)

    def _inject_implicit_mixed_number(self, step: str) -> str:
        """
        Automatically make a mixed number evalable
        e.g. 7 3/4 => 7+3/4
        """
        p1 = re.compile("([0-9]) +([0-9])")
        step = p1.sub("\\1+\\2", step)  ## implicit mults
        return step

    def _strip_properly_formatted_commas(self, expr: str) -> str:
        # We want to be careful because we don't want to strip tuple commas
        p1 = re.compile(r"(\d)(,)(\d\d\d)($|\D)")
        while True:
            next_expr = p1.sub("\\1\\3\\4", expr)
            if next_expr == expr:
                break
            expr = next_expr
        return expr

    def _grade_answer_sympy(self, given_answer: str, ground_truth: str) -> bool:
        ground_truth_normalized = self._normalize(ground_truth)
        given_normalized = self._normalize(given_answer)

        if ground_truth_normalized is None:
            return False

        if ground_truth_normalized == given_normalized:
            return True

        if len(given_normalized) == 0:
            return False

        ground_truth_elems = self._split_tuple(ground_truth_normalized)
        given_elems = self._split_tuple(given_normalized)

        if len(ground_truth_elems) > 1 and (
            ground_truth_normalized[0] != given_normalized[0]
            or ground_truth_normalized[-1] != given_normalized[-1]
        ):
            is_correct = False
        elif len(ground_truth_elems) != len(given_elems):
            is_correct = False
        else:
            for ground_truth_elem, given_elem in zip(ground_truth_elems, given_elems):
                if self._is_frac(ground_truth_elem) and self._is_frac(given_elem):
                    # if fractions aren't reduced, then shouldn't be marked as correct
                    # so, we don't want to allow sympy.simplify in this case
                    is_correct = ground_truth_elem == given_elem
                elif self._str_is_int(ground_truth_elem) != self._str_is_int(given_elem):
                    # if the ground truth answer is an integer, we require the given answer to be a strict match (no sympy.simplify)
                    is_correct = False
                else:
                    is_correct = self._are_equal_under_sympy(ground_truth_elem, given_elem)
                if not is_correct:
                    break

        return is_correct

    def _split_tuple(self, expr: str) -> List[str]:
        """
        Split the elements in a tuple/interval, while handling well-formatted commas in large numbers
        """
        expr = self._strip_properly_formatted_commas(expr)
        if len(expr) == 0:
            return []
        
        TUPLE_CHARS = "()[]"
        if (
            len(expr) > 2
            and expr[0] in TUPLE_CHARS
            and expr[-1] in TUPLE_CHARS
            and all([ch not in expr[1:-1] for ch in TUPLE_CHARS])
        ):
            elems = [elem.strip() for elem in expr[1:-1].split(",")]
        else:
            elems = [expr]
        return elems

    def _are_equal_under_sympy(self, ground_truth_normalized: str, given_normalized: str) -> bool:
        are_equal = False
        try:
            expr = f"({ground_truth_normalized})-({given_normalized})"
            if self._should_allow_eval(expr):
                sympy_diff = self._sympy_parse(expr)
                simplified = sympy.simplify(sympy_diff)
                if simplified == 0:
                    are_equal = True
        except:
            pass
        return are_equal

    def _should_allow_eval(self, expr: str) -> bool:
        # we don't want to try parsing unknown text or functions of more than two variables
        if self._count_unknown_letters_in_expr(expr) > 2:
            return False

        BAD_SUBSTRINGS = ["^{", "^("]
        BAD_REGEXES = [r"\^[0-9]+\^", r"\^[0-9][0-9]+"]
        
        for bad_string in BAD_SUBSTRINGS:
            if bad_string in expr:
                return False

        for bad_regex in BAD_REGEXES:
            if re.search(bad_regex, expr) is not None:
                return False

        return True

    def _count_unknown_letters_in_expr(self, expr: str) -> int:
        expr = expr.replace("sqrt", "")
        expr = expr.replace("frac", "")
        letters_in_expr = set([x for x in expr if x.isalpha()])
        return len(letters_in_expr)

    def _sympy_parse(self, expr: str):
        """Parses an expression with sympy."""
        py_expr = expr.replace("^", "**")
        return sympy_parser.parse_expr(
            py_expr,
            transformations=(
                sympy_parser.standard_transformations
                + (sympy_parser.implicit_multiplication_application,)
            ),
        )