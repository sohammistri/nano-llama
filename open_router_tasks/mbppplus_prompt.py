"""
3-shot prompt examples for MBPP+ evaluation.
Canonical few-shot examples matching the EvalPlus MBPP+ template.
Each tuple is (user_message, assistant_code_response).
"""

FEW_SHOT_EXAMPLES = [
    (
        # task_id=2: similar_elements
        "You are an expert Python programmer, and here is your task:\n"
        "Write a function to find the similar elements from the given two tuple lists.\n"
        "Your code should pass the following tests:\n"
        "assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)\n"
        "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)\n"
        "assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)",
        "```python\n"
        "def similar_elements(test_tup1, test_tup2):\n"
        "  res = tuple(set(test_tup1) & set(test_tup2))\n"
        "  return (res)\n"
        "```",
    ),
    (
        # task_id=3: is_not_prime
        "You are an expert Python programmer, and here is your task:\n"
        "Write a python function to identify non-prime numbers.\n"
        "Your code should pass the following tests:\n"
        "assert is_not_prime(2) == False\n"
        "assert is_not_prime(10) == True\n"
        "assert is_not_prime(35) == True",
        "```python\n"
        "import math\n"
        "def is_not_prime(n):\n"
        "    result = False\n"
        "    for i in range(2,int(math.sqrt(n)) + 1):\n"
        "        if n % i == 0:\n"
        "            result = True\n"
        "    return result\n"
        "```",
    ),
    (
        # task_id=4: heap_queue_largest
        "You are an expert Python programmer, and here is your task:\n"
        "Write a function to find the largest integers from a given list of numbers "
        "using heap queue algorithm.\n"
        "Your code should pass the following tests:\n"
        "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65] \n"
        "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75] \n"
        "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]",
        "```python\n"
        "import heapq as hq\n"
        "def heap_queue_largest(nums,n):\n"
        "    largest_nums = hq.nlargest(n, nums)\n"
        "    return largest_nums\n"
        "```",
    ),
]
