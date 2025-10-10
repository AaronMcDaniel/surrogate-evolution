"""
Tree Simplification Utility for Evolutionary Algorithm Individuals

This utility performs constant folding on tree-based representations of individuals
to reduce bloat by collapsing subtrees that contain only operations on constants.

Usage:
    from tree_simplifier import TreeSimplifier
    
    simplifier = TreeSimplifier()
    simplified_tree = simplifier.simplify(tree_string)
"""

import re
import copy
from typing import List, Any, Union
import primitives
from primitive_tree import CustomPrimitiveTree


class TreeSimplifier:
    """
    A utility class for simplifying evolutionary algorithm tree representations
    by performing constant folding.
    """
    
    def __init__(self):
        """Initialize the tree simplifier with primitive set information."""
        self.pset = primitives.pset
        
        # Cache for function evaluability to avoid repeated checks
        self._evaluability_cache = {}
    
    def _fold_constants_in_string(self, tree_string: str) -> str:
        """
        Fold constants in the tree string using a stack-based approach similar to codec.py
        
        Args:
            tree_string (str): The tree representation as a string
            
        Returns:
            str: The simplified tree string
        """
        # Tokenize similar to codec.py
        expr = re.split(r'([(),])', tree_string)
        remove = [',', '']
        expr = [x.strip() for x in expr if x.strip() not in remove]
        
        stack = []
        
        for element in expr:
            if element != ')':
                stack.append(element)
            else:
                # Collect arguments
                arguments = []
                while stack and stack[-1] != '(':
                    arguments.insert(0, stack.pop())
                
                if stack and stack[-1] == '(':
                    stack.pop()  # Remove the '('
                
                if stack:
                    function = stack.pop()
                    
                    # Try to evaluate this function call dynamically (like codec.py)
                    try:
                        # Parse arguments to their proper types
                        parsed_args = [self._parse_arg(x) for x in arguments]
                        
                        # Try to evaluate - if this succeeds, it's a mathematical function
                        result = eval(f'primitives.{function}({",".join([self._arg_to_eval_string(arg) for arg in parsed_args])})')
                        
                        # If the result is a simple type (int, float, bool, str), replace with the result  
                        if isinstance(result, (int, float, bool, str)):
                            stack.append(str(result))
                        else:
                            # If it's a complex object, this function shouldn't be folded
                            # Reconstruct the function call
                            reconstructed = f"{function}({', '.join(arguments)})"
                            stack.append(reconstructed)
                            
                    except Exception:
                        # This function couldn't be evaluated - it's likely a layer function
                        # Reconstruct the function call
                        reconstructed = f"{function}({', '.join(arguments)})"
                        stack.append(reconstructed)
        
        return stack[0] if stack else tree_string
    
    def _parse_arg(self, s):
        """Parse a string argument to its proper type, similar to codec.py"""
        try:
            return int(s)
        except ValueError:
            try:
                return float(s)
            except ValueError:
                # Handle boolean values
                if s.strip() == 'True':
                    return True
                elif s.strip() == 'False':
                    return False
                else:
                    return s
    
    def _arg_to_eval_string(self, arg):
        """Convert an argument to a string suitable for eval()"""
        if isinstance(arg, str) and arg not in ['True', 'False']:
            # Check if it's already a reconstructed function call
            if '(' in arg and ')' in arg:
                return arg
            else:
                return f'"{arg}"'
        else:
            return str(arg)

    def simplify(self, tree_string: str) -> str:
        """
        Simplify a tree string by performing constant folding.
        
        Args:
            tree_string (str): The tree representation as a string
            
        Returns:
            str: The simplified tree string
        """
        # Use a simpler approach similar to codec.py
        # Parse the expression and identify constant subtrees
        simplified = self._fold_constants_in_string(tree_string)
        return simplified
    
    def _tokenize(self, tree_string: str) -> List[str]:
        """
        Tokenize the tree string into individual components.
        
        Args:
            tree_string (str): The tree representation as a string
            
        Returns:
            List[str]: List of tokens
        """
        # Split on parentheses and commas, but keep them as separate tokens
        tokens = re.split(r'([(),])', tree_string)
        # Remove empty strings and whitespace
        tokens = [token.strip() for token in tokens if token.strip() and token.strip() != '']
        return tokens
    
    def _parse_tokens(self, tokens: List[str]) -> dict:
        """
        Parse tokens into a tree structure.
        
        Args:
            tokens (List[str]): List of tokens
            
        Returns:
            dict: Tree structure representation
        """
        stack = []
        i = 0
        
        while i < len(tokens):
            token = tokens[i]
            
            if token == '(':
                # Start of arguments for the previous function
                stack.append('(')
            elif token == ')':
                # End of arguments - collect all arguments for this function
                args = []
                while stack and stack[-1] != '(':
                    args.insert(0, stack.pop())
                
                if stack and stack[-1] == '(':
                    stack.pop()  # Remove the '('
                
                if stack:
                    func = stack.pop()
                    # Create a function node
                    node = {
                        'type': 'function',
                        'name': func,
                        'args': args
                    }
                    stack.append(node)
            elif token == ',':
                # Separator, ignore
                pass
            else:
                # This is either a function name or a constant
                if i + 1 < len(tokens) and tokens[i + 1] == '(':
                    # This is a function name
                    stack.append(token)
                else:
                    # This is a constant
                    try:
                        # Try to parse as a number
                        if '.' in token:
                            value = float(token)
                        else:
                            value = int(token)
                    except ValueError:
                        # It's a string constant (like 'True', 'False', or enum values)
                        if token == 'True':
                            value = True
                        elif token == 'False':
                            value = False
                        else:
                            value = token
                    
                    node = {
                        'type': 'constant',
                        'value': value
                    }
                    stack.append(node)
            
            i += 1
        
        return stack[0] if stack else None
    
    def _fold_constants(self, node: dict) -> dict:
        """
        Perform constant folding on the tree.
        
        Args:
            node (dict): Tree node
            
        Returns:
            dict: Simplified tree node
        """
        if node is None:
            return None
        
        if node['type'] == 'constant':
            return node
        
        if node['type'] == 'function':
            func_name = node['name']
            
            # First, recursively fold constants in arguments
            folded_args = []
            for arg in node['args']:
                if isinstance(arg, dict):
                    folded_args.append(self._fold_constants(arg))
                else:
                    folded_args.append(arg)
            
            # Check if all arguments are constants first (optimization)
            all_constants = True
            constant_values = []
            
            for arg in folded_args:
                if isinstance(arg, dict):
                    if arg['type'] == 'constant':
                        constant_values.append(arg['value'])
                    else:
                        all_constants = False
                        break
                else:
                    # Direct value
                    constant_values.append(arg)
            
            # Only try to fold if all arguments are constants
            if all_constants:
                # Try to dynamically evaluate this function call
                # Using the same approach as codec.py
                try:
                    result = self._evaluate_function_dynamically(func_name, constant_values)
                    return {
                        'type': 'constant',
                        'value': result
                    }
                except Exception:
                    # If evaluation fails, this is not a foldable function
                    # Return the original node with folded args
                    pass
            
            # Return the node with folded arguments
            return {
                'type': 'function',
                'name': func_name,
                'args': folded_args
            }
        
        return node
    
    def _evaluate_function_dynamically(self, func_name: str, args: List[Any]) -> Any:
        """
        Dynamically evaluate a function with constant arguments using the same 
        approach as codec.py. This will only succeed for mathematical/computational
        functions, not for layer/architectural functions.
        
        Args:
            func_name (str): Name of the function
            args (List[Any]): List of constant arguments
            
        Returns:
            Any: Result of the function evaluation
            
        Raises:
            Exception: If the function cannot be evaluated (e.g., it's a layer function)
        """
        # Check cache first
        cache_key = (func_name, tuple(str(arg) for arg in args))
        if cache_key in self._evaluability_cache:
            if self._evaluability_cache[cache_key] is None:
                raise ValueError(f"Function {func_name} is not evaluable (cached)")
            return self._evaluability_cache[cache_key]
        
        try:
            # Convert arguments to strings for the eval call, similar to codec.py
            arg_strings = []
            for arg in args:
                if isinstance(arg, str):
                    arg_strings.append(f'"{arg}"')
                else:
                    arg_strings.append(str(arg))
            
            # Use eval to dynamically call the function, just like codec.py does
            result_str = str(eval(f'primitives.{func_name}({",".join(arg_strings)})'))
            
            # Try to parse the result back to a proper type
            try:
                if '.' in result_str:
                    result = float(result_str)
                else:
                    result = int(result_str)
            except ValueError:
                # Handle boolean and other string results
                if result_str == 'True':
                    result = True
                elif result_str == 'False':
                    result = False
                else:
                    result = result_str
            
            # Cache the successful result
            self._evaluability_cache[cache_key] = result
            return result
            
        except Exception as e:
            # Cache that this function is not evaluable
            self._evaluability_cache[cache_key] = None
            raise e
    
    def _parse_arg(self, s):
        """
        Parse argument string to appropriate type, similar to codec.py __parse_arg method.
        
        Args:
            s: Argument string
            
        Returns:
            Parsed argument value
        """
        try:
            return int(s)
        except ValueError:
            try:
                return float(s)
            except ValueError:
                return s
    
    def _tree_to_string(self, node: dict) -> str:
        """
        Convert a tree structure back to string representation.
        
        Args:
            node (dict): Tree node
            
        Returns:
            str: String representation of the tree
        """
        if node is None:
            return ""
        
        if node['type'] == 'constant':
            value = node['value']
            if isinstance(value, bool):
                return str(value)
            elif isinstance(value, (int, float)):
                return str(value)
            else:
                return str(value)
        
        elif node['type'] == 'function':
            func_name = node['name']
            if not node['args']:
                return f"{func_name}()"
            
            arg_strings = []
            for arg in node['args']:
                if isinstance(arg, dict):
                    arg_strings.append(self._tree_to_string(arg))
                else:
                    arg_strings.append(str(arg))
            
            args_str = ", ".join(arg_strings)
            return f"{func_name}({args_str})"
        
        return str(node)
    
    def get_complexity_reduction(self, original_tree: str, simplified_tree: str) -> dict:
        """
        Calculate the complexity reduction achieved by simplification.
        
        Args:
            original_tree (str): Original tree string
            simplified_tree (str): Simplified tree string
            
        Returns:
            dict: Statistics about the complexity reduction
        """
        orig_tokens = self._tokenize(original_tree)
        simp_tokens = self._tokenize(simplified_tree)
        
        # Count function calls
        orig_funcs = len([t for t in orig_tokens if t not in ['(', ')', ',']])
        simp_funcs = len([t for t in simp_tokens if t not in ['(', ')', ',']])
        
        return {
            'original_length': len(original_tree),
            'simplified_length': len(simplified_tree),
            'length_reduction': len(original_tree) - len(simplified_tree),
            'length_reduction_percent': ((len(original_tree) - len(simplified_tree)) / len(original_tree)) * 100,
            'original_tokens': len(orig_tokens),
            'simplified_tokens': len(simp_tokens),
            'token_reduction': len(orig_tokens) - len(simp_tokens),
            'token_reduction_percent': ((len(orig_tokens) - len(simp_tokens)) / len(orig_tokens)) * 100,
            'original_functions': orig_funcs,
            'simplified_functions': simp_funcs,
            'function_reduction': orig_funcs - simp_funcs,
            'function_reduction_percent': ((orig_funcs - simp_funcs) / orig_funcs) * 100 if orig_funcs > 0 else 0
        }


def simplify_tree(tree_string: str) -> str:
    """
    Convenience function to simplify a tree string.
    
    Args:
        tree_string (str): The tree representation as a string
        
    Returns:
        str: The simplified tree string
    """
    simplifier = TreeSimplifier()
    return simplifier.simplify(tree_string)


if __name__ == "__main__":
    # Test with the provided sample
    sample_tree = """FCOS_Head(LazyConvTranspose2d(ResNeXt(ConvNeXt(DenseNet(ConvNeXt(ReLU_2D(MobileNet_V3(IN0, 1, 1)), dummyOp(dummyOp(0)), dummyOp(dummyOp(1))), dummyOp(dummyOp(dummyOp(1))), dummyOp(dummyOp(dummyOp(0)))), dummyOp(dummyOp(dummyOp(dummyOp(2)))), dummyOp(dummyOp(dummyOp(dummyOp(1))))), dummyOp(dummyOp(dummyOp(dummyOp(dummyOp(0))))), dummyOp(dummyOp(dummyOp(dummyOp(dummyOp(2)))))), toChannel(mul(protectedSub(protectedSub(add(protectedSub(64, 96), protectedDiv(39, 1)), protectedDiv(mul(68, 43), protectedSub(18, 96))), protectedSub(protectedDiv(mul(4, 81), protectedSub(46, 87)), protectedSub(mul(2, 73), protectedSub(23, 9)))), protectedDiv(protectedSub(protectedDiv(add(21, 8), protectedDiv(34, 55)), mul(mul(39, 63), add(25, 49))), protectedDiv(protectedDiv(add(45, 72), mul(3, 60)), protectedSub(add(72, 46), add(24, 85)))))), toKernel(add(mul(add(mul(protectedDiv(97, 52), protectedSub(51, 65)), add(protectedSub(17, 90), add(71, 23))), mul(add(add(33, 33), mul(100, 66)), protectedDiv(protectedSub(70, 4), protectedSub(96, 9)))), protectedSub(protectedDiv(protectedSub(protectedDiv(90, 83), protectedSub(98, 69)), add(mul(99, 13), protectedDiv(72, 42))), protectedDiv(mul(add(40, 23), protectedSub(43, 5)), protectedDiv(add(49, 76), protectedSub(9, 6)))))), toKernel(mul(mul(mul(protectedDiv(protectedDiv(3, 33), protectedSub(8, 12)), add(add(2, 51), add(58, 60))), add(add(protectedDiv(3, 10), mul(12, 4)), protectedSub(mul(0, 43), protectedSub(20, 99)))), protectedSub(protectedSub(mul(add(80, 5), protectedSub(81, 42)), protectedSub(protectedSub(28, 53), protectedDiv(72, 49))), add(protectedSub(add(91, 19), add(41, 4)), add(add(99, 86), protectedDiv(53, 72)))))), toStride(protectedSub(protectedSub(mul(protectedSub(protectedDiv(16, 72), mul(55, 31)), protectedDiv(protectedDiv(50, 82), protectedDiv(74, 32))), mul(mul(add(30, 5), mul(80, 68)), add(protectedSub(3, 93), protectedDiv(13, 77)))), protectedDiv(protectedSub(protectedSub(mul(52, 75), mul(47, 18)), mul(protectedDiv(25, 22), mul(48, 37))), protectedDiv(add(protectedDiv(48, 44), protectedSub(14, 21)), add(protectedDiv(59, 82), add(69, 21)))))), toStride(protectedSub(protectedSub(protectedSub(add(protectedSub(45, 91), mul(52, 20)), mul(mul(48, 30), protectedDiv(49, 65))), mul(protectedDiv(protectedDiv(87, 18), add(89, 76)), add(mul(48, 4), add(46, 40)))), add(protectedDiv(mul(mul(99, 20), add(3, 66)), protectedDiv(protectedSub(90, 33), add(49, 56))), add(protectedDiv(add(42, 64), protectedDiv(61, 34)), protectedDiv(protectedDiv(33, 29), mul(38, 58)))))), toPadding(add(protectedDiv(protectedDiv(protectedDiv(add(61, 14), mul(46, 17)), add(protectedSub(0, 51), protectedDiv(60, 99))), mul(add(add(29, 39), mul(1, 45)), mul(add(39, 28), mul(97, 29)))), mul(protectedSub(protectedDiv(mul(49, 99), add(45, 29)), mul(mul(35, 51), mul(61, 15))), protectedDiv(protectedSub(mul(95, 43), protectedDiv(14, 49)), protectedDiv(protectedDiv(86, 64), mul(29, 66)))))), toPadding(add(protectedDiv(mul(protectedDiv(protectedSub(6, 52), add(25, 22)), add(protectedSub(3, 5), add(68, 16))), add(protectedSub(add(25, 66), protectedDiv(68, 29)), protectedSub(protectedDiv(60, 55), protectedSub(97, 78)))), mul(add(mul(mul(86, 82), protectedSub(32, 91)), add(mul(88, 79), protectedSub(26, 25))), protectedDiv(add(add(15, 60), protectedSub(70, 16)), protectedDiv(mul(10, 29), protectedDiv(44, 99)))))), dummyOp(dummyOp(dummyOp(dummyOp(dummyOp(dummyOp(3)))))), toDilation(mul(protectedDiv(protectedDiv(protectedSub(protectedSub(36, 83), protectedDiv(66, 24)), mul(add(19, 4), mul(98, 49))), protectedSub(protectedDiv(protectedDiv(21, 6), mul(92, 23)), protectedDiv(protectedSub(84, 76), protectedDiv(31, 77)))), mul(protectedSub(add(add(17, 56), protectedSub(96, 82)), mul(mul(87, 20), mul(91, 50))), protectedDiv(mul(protectedDiv(18, 96), add(50, 90)), protectedSub(mul(2, 26), add(74, 69)))))), toDilation(add(protectedSub(protectedDiv(mul(protectedDiv(10, 96), add(53, 39)), add(add(78, 75), mul(59, 14))), protectedDiv(mul(protectedSub(49, 29), mul(5, 40)), protectedDiv(add(55, 49), add(55, 97)))), protectedSub(add(add(protectedDiv(4, 33), protectedDiv(44, 47)), protectedSub(add(65, 20), protectedDiv(81, 42))), protectedSub(mul(add(97, 32), add(75, 62)), add(protectedSub(12, 22), protectedSub(25, 44)))))), toGroup(mul(mul(protectedSub(protectedSub(protectedDiv(16, 71), add(84, 39)), mul(add(78, 37), add(91, 25))), mul(protectedDiv(add(64, 15), add(41, 37)), protectedSub(protectedDiv(4, 1), mul(84, 40)))), mul(add(protectedSub(protectedSub(76, 100), protectedDiv(35, 32)), mul(protectedDiv(90, 54), mul(66, 29))), protectedDiv(protectedDiv(add(10, 12), protectedSub(56, 96)), mul(protectedDiv(67, 63), mul(51, 60))))))), SGD(protectedSub(toProbFloat(toBoundedFloat(toPNorm(mul(protectedSub(15.115038470408681, 78.88696130014806), protectedSub(54.41694261673952, 23.263343322418596))))), toBoundedFloat(toProbFloat(protectedDiv(toProbFloat(protectedSub(75.82587417958115, 88.10669096802846)), toPNorm(toProbFloat(74.49870575066898)))))), toPNorm(toBoundedFloat(mul(toBoundedFloat(toBoundedFloat(toBoundedFloat(0.7219422568446254))), toBoundedFloat(toBoundedFloat(protectedSub(0.41204101717330377, 2.0110586492404856)))))), protectedSub(protectedDiv(toPNorm(toPNorm(protectedDiv(add(85.58048165288653, 0.8991226391869495), add(0.9615653287386672, 74.56964432885293)))), protectedDiv(toProbFloat(mul(protectedDiv(66.18373349032933, 72.61957829939958), toBoundedFloat(58.6031505783428))), toProbFloat(protectedSub(toBoundedFloat(0.7616061557397289), add(0.693122442411298, 45.06107919673597))))), toBoundedFloat(toBoundedFloat(toProbFloat(protectedSub(toPNorm(1.349177657340206), protectedSub(66.68484237639565, 82.96932873404428)))))), mul(toPNorm(toProbFloat(toProbFloat(mul(protectedSub(0.47944845860386953, 60.85645213492715), mul(1.602839401965922, 47.544167055726895))))), mul(toBoundedFloat(toProbFloat(mul(mul(1.7917236771993634, 2.391077889432417), toPNorm(67.45820702230903)))), toBoundedFloat(toPNorm(toBoundedFloat(protectedSub(43.13239247827596, 51.43713516549059))))))), LinearLR(protectedSub(toBoundedFloat(toPNorm(toPNorm(toBoundedFloat(toPNorm(2.8341374299185746))))), mul(protectedDiv(mul(protectedDiv(toProbFloat(0.7894576618783822), protectedSub(45.88175691021558, 0.018278577933135476)), toPNorm(add(97.8997474863433, 0.8254610141633401))), protectedSub(protectedSub(toPNorm(90.72143134904128), toProbFloat(60.93977899236701)), mul(add(0.020650972482342933, 83.12608589988236), mul(39.30476369202842, 2.4766779000055488)))), toBoundedFloat(mul(add(toPNorm(2.564135970403865), toProbFloat(0.7680779559328453)), add(toPNorm(35.278860284662805), protectedSub(1.3936160999427487, 79.5624644896493)))))), protectedDiv(toProbFloat(protectedSub(toPNorm(toPNorm(toPNorm(2.1736035142462136))), toBoundedFloat(protectedSub(add(0.43613434506515236, 0.646381919440947), toProbFloat(50.246342335058856))))), protectedDiv(mul(mul(toProbFloat(toProbFloat(71.44289840866817)), mul(protectedSub(97.9000978023168, 48.60464928826954), protectedDiv(26.957464140514276, 0.458658291430426))), toBoundedFloat(protectedSub(add(55.89262002321001, 2.216016597267858), protectedDiv(60.747744086199894, 2.5325703490734615)))), toProbFloat(toProbFloat(toBoundedFloat(toBoundedFloat(0.5845260007184313)))))), protectedSub(protectedSub(protectedDiv(mul(add(protectedDiv(35, 81), mul(1, 77)), add(protectedSub(52, 100), protectedDiv(40, 10))), protectedDiv(protectedSub(add(52, 6), protectedDiv(71, 52)), protectedDiv(add(69, 71), protectedSub(75, 11)))), protectedDiv(mul(mul(mul(51, 81), add(1, 19)), protectedSub(protectedSub(42, 94), mul(42, 11))), mul(protectedSub(mul(42, 60), add(40, 49)), protectedDiv(protectedDiv(66, 28), protectedSub(65, 18))))), add(protectedSub(protectedSub(add(protectedDiv(85, 67), mul(45, 51)), mul(protectedDiv(29, 92), protectedDiv(43, 31))), add(protectedDiv(add(55, 69), add(5, 5)), mul(protectedSub(34, 61), protectedDiv(82, 83)))), protectedDiv(mul(protectedSub(add(67, 49), mul(27, 90)), protectedDiv(protectedSub(40, 55), protectedSub(50, 47))), protectedDiv(mul(protectedSub(32, 68), mul(30, 79)), protectedSub(mul(18, 70), protectedSub(57, 26))))))), toProbFloat(toPNorm(mul(toPNorm(protectedSub(toProbFloat(toBoundedFloat(0.5408181361610681)), protectedSub(mul(1.5468639658651617, 35.3948636681213), protectedDiv(73.08698223586157, 0.1920902874812006)))), protectedSub(toPNorm(toBoundedFloat(protectedSub(1.1618711240716946, 68.86314264419748))), protectedDiv(toBoundedFloat(mul(1.1089050600439643, 0.8925512437427088)), mul(protectedSub(61.498842924984075, 1.617757894813368), protectedDiv(1.749951842730981, 0.7261892414106231))))))), toProbFloat(toPNorm(add(toBoundedFloat(toProbFloat(toPNorm(toPNorm(95.56997323999383)))), toBoundedFloat(toPNorm(add(toPNorm(2.019692988134558), toBoundedFloat(0.8235857556247053))))))), toProbFloat(add(toBoundedFloat(toPNorm(toBoundedFloat(protectedDiv(toProbFloat(77.21675700850072), toPNorm(27.77233816346819))))), protectedSub(protectedSub(toPNorm(protectedSub(protectedSub(87.59271743022794, 49.60068821048771), mul(73.44217420832106, 20.443641452817317))), protectedSub(toBoundedFloat(toProbFloat(2.9481285803356565)), protectedSub(protectedSub(0.3514486814873079, 2.356206180545634), toBoundedFloat(8.156322612191847)))), toProbFloat(toProbFloat(toPNorm(toBoundedFloat(79.17829796606796))))))), toProbFloat(add(protectedSub(toBoundedFloat(protectedSub(toProbFloat(toPNorm(0.4181974539693931)), protectedSub(protectedDiv(0.5033208304550228, 66.98096333027411), protectedDiv(66.4163312921053, 1.4628853193018034)))), toProbFloat(toProbFloat(toProbFloat(mul(10.03831353023702, 1.5443140767905907))))), toPNorm(toPNorm(mul(protectedSub(mul(1.1309532698273772, 1.2275903814162752), toPNorm(0.5169079739788612)), protectedSub(toProbFloat(1.2883609596275538), mul(76.13674982733806, 73.9064835521827))))))), toProbFloat(protectedDiv(mul(protectedDiv(toPNorm(add(mul(2.4712098795463495, 2.0991554279009472), toBoundedFloat(65.60632589070853))), mul(toPNorm(protectedDiv(1.4202691237436509, 39.142621300972756)), add(add(1.2432084606098912, 2.4988363482571714), protectedSub(0.20151169771708288, 9.513733236871747)))), protectedDiv(protectedDiv(add(toBoundedFloat(52.56362721227988), protectedSub(1.1817145078102493, 0.5132036218808965)), toPNorm(protectedSub(2.2015142201617, 67.26548433337393))), protectedDiv(toProbFloat(toProbFloat(0.3507921572430006)), toBoundedFloat(toBoundedFloat(70.20554739571674))))), add(protectedSub(protectedDiv(protectedSub(toPNorm(82.56735802219566), protectedSub(45.37412708389756, 0.6042713692491128)), toBoundedFloat(toProbFloat(39.2182859249744))), toProbFloat(toProbFloat(toProbFloat(87.72847627912127)))), add(toPNorm(add(toBoundedFloat(1.7411378266305415), toPNorm(0.23009749398538937))), protectedSub(protectedDiv(protectedSub(0.48492025794768734, 51.7199924920388), toProbFloat(73.92090236929509)), protectedSub(add(32.01941451689181, 0.359494230923597), toBoundedFloat(20.022937651501593))))))), toProbFloat(toPNorm(toProbFloat(protectedDiv(toProbFloat(toProbFloat(toProbFloat(1.7895225738805354))), protectedSub(protectedDiv(add(62.94496332199249, 52.671457222126286), toBoundedFloat(1.699846314718513)), toPNorm(protectedSub(1.529747973994346, 11.677425311343903))))))), toProbFloat(protectedSub(protectedDiv(toBoundedFloat(toPNorm(protectedDiv(protectedSub(0.29887829534749877, 2.1457959873271433), toProbFloat(64.73867664242333)))), add(add(protectedDiv(protectedSub(0.030108130431385716, 0.46565282956871523), mul(70.27155798859228, 1.3249483739970553)), toProbFloat(add(95.46400643636818, 1.3157764274433492))), protectedSub(toProbFloat(toPNorm(86.4169177687381)), protectedSub(toBoundedFloat(6.791389812235494), mul(0.26953129189431313, 2.1543909427533317))))), mul(toPNorm(protectedDiv(add(toProbFloat(0.1413590671681363), protectedDiv(0.3470377201799839, 63.725127583588495)), protectedDiv(add(65.29062110324702, 77.61490351643135), toPNorm(24.78571748182603)))), toPNorm(protectedSub(protectedDiv(mul(0.7652163598722272, 2.156481482375737), toPNorm(71.0806357353092)), toProbFloat(toPNorm(2.848017859465379))))))))"""
    
    simplifier = TreeSimplifier()
    
    print("Original tree:")
    print(sample_tree)
    print("\nSimplifying...")
    
    try:
        simplified = simplifier.simplify(sample_tree)
        print("\nSimplified tree:")
        print(simplified)
        
        stats = simplifier.get_complexity_reduction(sample_tree, simplified)
        print("\nComplexity reduction statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"Error during simplification: {e}")
        import traceback
        traceback.print_exc()