from concurrent.futures import ThreadPoolExecutor, TimeoutError
import numpy as np
import math
import re
from datetime import datetime
import json
from tracr.rasp import rasp
from tracr.compiler import compiling
from tracr.compiler.validating import dynamic_validate
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from collections import Counter
import logging
from tqdm import tqdm
from typing import Optional, List, Any
from tracr.compiler.assemble import AssembledTransformerModel

def colored(r: int, g: int, b: int, text: str):
    """
    Color the text for better readability in the console output

    Parameters:
    - r: int, red in the rgb mix
    - g: int, green in the rgb mix
    - b: int, blue in the rgb mix

    Returns:
    The colored string, ready to be printeded: str
    """
    return "\033[38;2;{};{};{}m{}\033[0m".format(r, g, b, text)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='process_task_log.txt', filemode='w')
# To also logging.info to console, add a StreamHandler
console = logging.StreamHandler()
console.setLevel(logging.INFO)  # Change this to DEBUG or INFO for more verbose output in the console
formatter = logging.Formatter(colored(255,255,255,f'%(message)s'))
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)




def replace_task_lines(file_path: str, task: str, function_name: str):
    """
    Replace task lines in a prompt file.
    Parameters:
    - file_path: str, path to the prompt file.
    - task: str, task description to be added.
    - function_name: str, name that the generated function should have.
    Returns:
    None
    """
    # Extract the example from the task
    example_start = task.find("Example: ")
    if example_start != -1:
        example = task[example_start:]
        task = task[:example_start].strip()
    else:
        example = ""

    # Format the task and example
    task_lines = [
        "# Your Task",
        f"{task}",
        f"{example}",
        f"Name the function that you can call to make this program '{function_name}()'"
    ]

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='ISO-8859-1') as file:  # Try a different encoding
            lines = file.readlines()

    with open(file_path, 'w', encoding='utf-8') as file:
        task_section_started = False
        task_section_ended = False
        for line in lines:
            if line.startswith("# Your Task"):
                task_section_started = True
                task_section_ended = False
                file.write("\n".join(task_lines) + "\n")
            elif task_section_started and not task_section_ended:
                if line.strip() == "":
                    task_section_ended = True
                    file.write(line)
            else:
                file.write(line)

def extract_python_code(model_output: str):
    """
    Take the model output and extract the python in the code block.

    Parameters:
    - model_output: str, the whole output of a model (should contain a code block denoted with ```python(.*?)```).

    Returns:
    The contents of the code block: str
    """
    # Regex pattern to match Python code blocks
    pattern = r'```python(.*?)```'
    # Using re.DOTALL to make '.' match newlines as well
    matches = re.findall(pattern, model_output, re.DOTALL)
    if matches:
        # If there are two or more matches, return the second one; otherwise, return the first one
        return matches[-1].strip().replace("import rasp", "") if len(matches) >= 2 else matches[0].strip().replace("import rasp", "")
    else:
        raise ValueError("No Python code block found.")





# execute the produced rasp code
def check_op_runs(op: rasp.SOp, verbose=True):
    """
    Test whether a rasp operation actually runs on a generic input or whether the function already throws errors

    Parameters:
    - op: rasp.SOp, rasp program to be tested
    - verbose: boolean = True, determines whether the results are logging.infoed

    Returns:
    The rasp operation: rasp.SOp
    """
    input = [0, 3, 4, 1, -5, 4]  # this is a test array chosen because it contains 0, negative and positive elements and duplicats
    result = op(input)
    if verbose:
        logging.info("the function runs correctly."+ str(input) + "-->"+str(result))
    return op


def are_arrays_almost_equal(array1, array2, tolerance=1e-10):
    """
    Check if two arrays are almost equal within a specified tolerance. Both arrays are compared element by element. If all corresponding pairs of elements are within the given tolerance, the arrays are considered almost equal.

    Parameters:
    - array1: list, first array to be compared.
    - array2: list, second array to be compared.
    - tolerance: float = 1e-10, the tolerance level for comparing elements. Both relative and absolute tolerances are considered.

    Returns:
    - bool: True if arrays are almost equal within the tolerance, otherwise False.
    """
    if array1 == array2:
      return True
    if len(array1) != len(array2):
        return False
    for a, b in zip(array1, array2):
        if a is None and b is None:
            continue
        if a is None or b is None:
            return False
        if not math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance):
            return False
    return True


def check_op_with_ground_truth(op: rasp.SOp, function, tests: Optional[List[Any]] = None, verbose=True):
    """
    Test whether a rasp operation computes the correct output based on 1000 examples. Throws errors if not

    Parameters:
    - op: rasp.SOp, rasp program to be tested
    - fun: pyhton function, ground truth function that is used to generate the ground truth input-output pairs for the rasp op to be evaluated against
    - tests: list = None, a list where every entry is a dict with keys "input", "output" which each point to a list of integers. This pair is supposed to represent one example of the ground truth input output pair
    - verbose: boolean = True, determines whether the results are logging.infoed

    Returns:
    None
    """
    ground_truth_function = function
    all_correct = True
    n_errors = 0
    if tests is None:
        for i in range(1000):
            rand_arr = [np.random.randint(10) for _ in range((np.random.randint(5) + 1) * 2)]
            rasp_result = op(rand_arr)
            ground_truth = ground_truth_function(rand_arr)
            if not are_arrays_almost_equal(rasp_result,ground_truth):
                    all_correct = False
                    n_errors += 1
    else:
        for test in tests:
            rand_arr = test["input"]
            ground_truth = test["output"]
            rasp_result = op(rand_arr)
            if not are_arrays_almost_equal(rasp_result,ground_truth):
                all_correct = False
                n_errors += 1
    if all_correct:
        if verbose:
            logging.info("the rasp program is ground truth equivalent")
    else:
        error_rate = n_errors / 1000 if tests is None else n_errors / len(tests)
        if verbose:
            assert all_correct, f"the rasp program doesn't produce the correct output (for input [3,8,2,1,5,4]) --> {str(op([3, 8, 2, 1, 5, 4]))}. ({error_rate} error rate)"
        else:
            assert all_correct


def check_op_with_validator(op, tests: Optional[List[Any]] = None, verbose=True):
    """
    Test whether a rasp operation passes the validator that's integrated in tracr. This is meant to determine whether the program is fit to be compiled or whether the resulting weights will be erronious.

    Parameters:
    - op: rasp.SOp, rasp program to be tested
    - tests: list = None, a list where every entry is a dict with keys "input", "output" which each point to a list of integers. The input is used by the validator.
    - verbose: boolean = True, determines whether the results are printed

    Returns:
    None
    """
    issues = []
    if tests is None:
        for i in range(1000):
            rand_arr = [np.random.randint(10) for _ in range((np.random.randint(5) + 1) * 2)]
            issues += dynamic_validate(op, rand_arr)
    else:
        for test in tests:
            rand_arr = test["input"]
            issues += dynamic_validate(op, rand_arr)
    if verbose:
        logging.info("number of issues found by the validator:"+str(len(issues)))
        assert len(issues) == 0, f"the following issue(s) were found by the validator: {issues[0]}"
    else:
        assert len(issues) == 0


def check_compileability(op: rasp.SOp, verbose=True):
    """
    Test whether a rasp operation compiles into tracr-model-weights without errors.

    Parameters:
    - op: rasp.SOp, rasp program to be tested
    - verbose: boolean = True, determines whether the results are printed

    Returns:
    model: AssembledTransformerModel, the model weights of the tracr transfrormer
    """
    try:
        voc = {i for i in range(10)}
        model = compiling.compile_rasp_to_model(
            op,
            vocab=voc,
            max_seq_len=10,
            compiler_bos="BOS",
            mlp_exactness=1000,
        )
        if verbose:
            logging.info("the model compiled correctly")
        return model
    except Exception as e:
        if verbose:
            logging.info("the program did not compile correctly. This was the exception:\n"+str(e))
        raise e



system_prompt = "As a skilled mathematician turned RASP programmer, your expertise lies in algorithm design and implementation. Your role involves crafting programs in RASP, a programming language with distinct characteristics. Approach each task methodically, breaking down problems into sequential steps. Pay close attention to the unique syntax and structure of RASP, to ensure precise and effective programming solutions. Try to keep it simple and go for the most straightfroward solution. Your critical thinking and step-by-step reasoning are essential in navigating the challenges posed by this unique programming environment."
#alternate system promt: generated by claude
#system_prompt = "As an expert RASP programmer and mathematician, your role is to design and implement efficient algorithms using the unique syntax and structure of the RASP language. Approach each task methodically, breaking down problems into clear, sequential steps. Pay close attention to RASP-specific features and aim for simplicity and clarity in your code. Use your strong analytical skills to devise optimal solutions, considering input formats, edge cases, and time/space complexity. Provide well-formatted and thoroughly tested code, following the specified output format. Your precision, attention to detail, and dedication to crafting high-quality RASP solutions are highly valued."
stages = {0: "defining op", 1: "testing function correctness", 2: "testing with validator", 3: "testing compileability", 4: "testing model correctness", 5: "success"}



def check_weight_correctness(weights: AssembledTransformerModel, op: rasp.SOp, tests: Optional[List[Any]] = None, verbose=True):
    """
    Test whether the tracr transformer produces the same output as the rasp operation it was compiled from

    Parameters:
    - weights: AssembledTransformerModel
    - op: rasp.SOp, rasp program to be tested
    - tests: list = None, a list where every entry is a dict with keys "input", "output" which each point to a list of integers. The input is used by the validator.
    - verbose: boolean = True, determines whether the results are logging.infoed

    Returns:
    None
    """
    weights_equivalent = True
    if tests is None:
        for i in range(100):
            rand_arr = [np.random.randint(5) for i in range((np.random.randint(3) + 1) * 2)]
            example_input = rand_arr
            ground_truth = op(example_input)
            if isinstance(ground_truth, float) or isinstance(ground_truth, int):
                ground_truth = [ground_truth for i in range(len(example_input))]
            ground_truth = ["BOS"] + ground_truth
            mo = weights.apply(["BOS"] + example_input).decoded
            if verbose:
                assert mo == ground_truth, "ground truth: " + str(ground_truth) + " model output: " + str(mo) + " for " + str(example_input)
            else:
                assert mo == ground_truth
            if mo != ground_truth:
                weights_equivalent = False
    else:
        for test in tests:
            rand_arr = test["input"]
            ground_truth = test["output"]
            example_input = rand_arr
            ground_truth = ["BOS"] + ground_truth
            mo = weights.apply(["BOS"] + example_input).decoded
            if verbose:
                assert mo == ground_truth, "ground truth: " + str(ground_truth) + " model output: " + str(mo) + " for " + str(example_input)
            else:
                assert mo == ground_truth
    if weights_equivalent and verbose:
        logging.info("weights are equivalent to the rasp function")


def run_with_timeout(func, args=(), kwargs={}, timeout=None):
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except TimeoutError as e:
            raise e

def process_task(model: AssembledTransformerModel, task_data: dict, promt_file: str, tries:int=5, verbose:bool=True):
    """
    Prompt a model to generate the rasp program specefied in the task_data using the prompt in a txt file. It then checks the generated code for various flaws. If it finds a flaw it records it and attempts again. It repeats this by default 5 times (as defined by the tries variable) and returns the results if no flaws are found or the 5th try is unsuccessful.

    Parameters:
    - model, must be a language model object with a .generate function. The generate function should take this as input: model.generate(prompt, system_prompt, "") and return the completed response as a string
    - task data: dict, should at least contain the keys:
        "instruction": points to a string with the format "[task description], Example[example of an input-output pair];[function name],
        "ground_truth_function": a python function implementing the task
        "test_list": a list of dicts, each with the keys "input" and "output" representing some inptu output pair fitting the task.
    - prompt_file:str, the path to the txt file containing the prompt.
    - tries: int = 5, the number of attempts the model has to get it right (default is 5)
    - verbose: boolean = True, determines whether the results are printed

    Returns:
    (successes, failures): list of dicts, each dict in the failures list has keys "generated rasp code", "feilure stage", "error", successes is either empty or contains one entry which is the successful rasp code as a string
    """
    successes = []
    failures = []
    task = task_data["instruction"].split(";")[0]
    function_name = task_data["instruction"].split(";")[1].replace("\n", "").replace(" ","")
    if verbose:
        logging.info("\n"+"-"* 180)
        logging.info("\nTask: "+ task)
        logging.info("Function Name: "+ function_name)
        logging.info("Modifying prompt")
    replace_task_lines(promt_file, task, function_name)
    generated_rasp_code = ""
    for i in range(tries):
        if verbose:
            logging.info(f"Attempt {i + 1}")
        stage = 0
        try:
            if verbose: logging.info("Generating RASP code")
            prompt = open(promt_file).read()
            output = model.generate(prompt, system_prompt, "")
            if verbose: logging.info("GENERATED CODE:")
            generated_rasp_code = extract_python_code(output)
            if verbose: logging.info(colored(0, 150, 200, generated_rasp_code))
            exec_environment = {}
            exec("from tracr.rasp import rasp\n"+generated_rasp_code, exec_environment)#+"\nop = " + function_name.replace(" ", "") + "()"+"\nlogging.info(op)")
            op = exec_environment[function_name.replace(" ", "")]()
            result = run_with_timeout(check_op_runs, args=(op,), kwargs={'verbose': verbose}, timeout=10)
            if result is None:
                raise Exception("test_run_op timed out")
            stage = 1
            if verbose: logging.info("\nGround truth function:")
            ground_truth_function = task_data["ground_truth_function"]
            if verbose: logging.info(colored(0, 150, 200, ground_truth_function))
            exec(ground_truth_function, exec_environment)
            if verbose:
                logging.info("TESTS:")
                logging.info("Testing against ground truth:")
            fun = exec_environment["fun"]
            check_op_with_ground_truth(op, fun, tests=task_data["test_lists"], verbose=verbose)

            stage = 2
            if verbose: logging.info("Testing with tracr validator:")
            check_op_with_validator(op, tests=task_data["test_lists"], verbose=verbose)
            stage = 3
            if verbose: logging.info("Testing compileability:")
            weights = check_compileability(op, verbose=verbose)
            stage = 4
            if verbose: logging.info("testing correctnes of the tracr transformer weights:")
            check_weight_correctness(weights, op, tests=task_data["test_lists"], verbose=verbose)
            stage = 5
            if verbose: logging.info(colored(0, 255, 0, f"Testing complete\nGenerated correct function after {i + 1} tries"))
            successes.append(generated_rasp_code)
            break
        except Exception as e:
            failures.append({"generated rasp code": generated_rasp_code, "failure stage": stage, "error": str(e)})
            if verbose: logging.info(colored(255, 0, 0, f"Failed at stage {stage} ({stages[stage]})\nError: " + str(e)))
            if i == tries - 1:
                if verbose: logging.info(f"failed to generate {function_name}")
    return successes, failures


def evaluate_model(model, verbose: bool = True):
    """
    Evaluate a language model on its ability to generate correct, compileable rasp code according to specification and save the results of the test to a json file.

    Parameters:
    - model, must be a language model object with a .generate function. The generate function should take this as input: model.generate(prompt, system_prompt, "") and return the completed response as a string
    - verbose: boolean = True, determines whether the results are logging.infoed

    Returns:
    results: dict, {"successes":successes, "failures":failures} successes is a dict of functional rasp strings where the function name like " make_sum_digits" is the key. Failures is also a dict with function names as key but each value is a list of dicts with this layout: {"generated rasp code": generated_rasp_code, "feilure stage": stage, "error": str(e)}
    """
    with open('eval_data.json', 'r') as file:
        data = json.load(file)
    suc, n_attempted_generations = 0, 0
    successes = {}
    failures = {}
    results = {"successes": successes, "failures": failures}
    promt_file = "prompts/prompt_full.txt"
    for key in  tqdm(list(data.keys())[34:], desc="Evaluating Tasks"):
        task_successes, task_failures = process_task(model, data[key], promt_file, verbose=verbose)
        n_attempted_generations += 1
        if len(task_successes) > 0:
            suc += 1
        successes[key] = task_successes
        failures[key] = task_failures
        if verbose:
            logging.info("\nSUCCESSRATE:" + str(suc) + "/"+str(n_attempted_generations)+ "\n")
        results = {"successes": successes, "failures": failures}
        with open(f'evaluation_results_{model.model_name}_{datetime.now().strftime("%Y%m%d")}.json', 'w') as f:
            json.dump(results, f, indent=4)
    return results


def visualize_results(results: dict):
    """
    Visualise the performance of a language model on the RASP eval, by showing how much it succeeded, and where it failed.

    Parameters:
    - results: dict, {"successes":successes, "failures":failures} successes is a dict of functional rasp strings where the function name like " make_sum_digits" is the key. Failures is also a dict with function names as key but each value is a list of dicts with this layout: {"generated rasp code": generated_rasp_code, "feilure stage": stage, "error": str(e)}

    Returns:
    None
    """
    stages = {0: "defining op", 1: "testing function correctness", 2: "testing with validator", 3: "testing compileability", 4: "testing model correctness", 5: "success"}
    max_fail_stage_for_tasks = {}
    for key in results["failures"].keys():
        if len(results["failures"][key]) > 4:
            max_fail_stage_for_tasks[key] = max([i["failure stage"] for i in results["failures"][key]])
        else:
            max_fail_stage_for_tasks[key] = 5
    stage_counts = {i: 0 for i in range(7)}
    for i in max_fail_stage_for_tasks.values():
        stage_counts[i] += 1
    test_arr = [stage_counts[i] for i in range(5, -1, -1)]
    cumulative_sum = sorted([sum(test_arr[:i + 1]) for i in range(len(test_arr))], reverse=True)
    colors = plt.cm.Blues(np.linspace(0.2, 1, len(cumulative_sum)))
    tries_per_success = {i: 0 for i in range(1, 6)}
    for key in results['successes'].keys():
        if len(results['successes'][key]) > 0:
            tries_per_success[len(results["failures"][key]) + 1] += 1
    tries = list(tries_per_success.keys())
    success_counts = list(tries_per_success.values())
    failure_stages = []
    for key in results["failures"].keys():
        for fail in results["failures"][key]:
            failure_stages.append(fail["failure stage"])
    failure_stage_counts = Counter(failure_stages)
    labels = [stages[key] for key in failure_stage_counts.keys()]
    sizes = list(failure_stage_counts.values())
    failure_colors = plt.cm.Blues(np.linspace(0.2, 1, len(labels)))

    # Create a new figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    # Plot 1: Cumulative Sum Bar Chart
    for i, val in enumerate(cumulative_sum):
        axs[0, 0].bar([0], val, color=colors[i], label=stages[i])
        height = val
        if i<len(cumulative_sum)-1:
            val -= cumulative_sum[i+1]
        axs[0, 0].text(0.5, height, f'{val}', va='center', ha='right', fontsize='small')
    axs[0, 0].legend(loc='lower right', fontsize='small', fancybox=True, framealpha=1)
    axs[0, 0].set_title('Cumulative Sum of Stages')
    # Plot 2: Successes by Number of Tries Bar Chart
    axs[0, 1].bar(tries, success_counts, color='darkblue')
    axs[0, 1].set_xlabel('Number of Tries')
    axs[0, 1].set_ylabel('Number of Successes')
    axs[0, 1].set_title('Successes by Number of Tries')
    axs[0, 1].set_xticks(tries)
    axs[0, 1].grid(axis='y')
    # Plot 3: Failure Stages Pie Chart
    axs[1, 0].pie(sizes, labels=labels, colors=failure_colors, autopct='%1.1f%%', startangle=140)
    axs[1, 0].axis('equal')
    axs[1, 0].set_title('Failure Stages Distribution')
    axs[1, 1].axis('off')
    # Plot 4: Ratio of correct to all tasks
    axs[1, 1].text(0.5, 0.5, 'Completed Tasks: '+str( len([key for key in results["successes"].keys() if len(results["successes"][key])>0]))+" / "+str(len(list(results["successes"].keys())))   ,
                   horizontalalignment='center', verticalalignment='center',
                   fontsize=16, transform=axs[1, 1].transAxes)
    # Adjust spacing between subplots
    plt.tight_layout()
    # Save the figure as a PDF file
    with PdfPages(f'visualization_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf') as pdf:
        pdf.savefig(fig)
    plt.show()