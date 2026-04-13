"""
4-shot prompt examples for MATH-500 evaluation.
From Minerva paper (arXiv 2206.14858), Section D.2.

Edit this file to experiment with different few-shot prompts.
Each tuple is (problem, solution) where solution ends with the
"Final Answer: The final answer is ..." format.
"""

FEW_SHOT_EXAMPLES = [
    (
        # Example 1: Precalculus / Domain
        r"Find the domain of the expression $\frac{\sqrt{x-2}}{\sqrt{5-x}}$.",
        r"""The expressions inside each square root must be non-negative. Therefore, $x-2 \ge 0$, so $x\ge2$, and $5 - x \ge 0$, so $x \le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\boxed{[2,5)}$.
Final Answer: The final answer is $\boxed{[2,5)}$. I hope it is correct.""",
    ),
    (
        # Example 2: Linear Algebra / Determinant
        r"If $\det \mathbf{A} = 2$ and $\det \mathbf{B} = 12$, then find $\det (\mathbf{A} \mathbf{B})$.",
        r"""We have that $\det (\mathbf{A} \mathbf{B}) = (\det \mathbf{A})(\det \mathbf{B}) = (2)(12) = \boxed{24}$.
Final Answer: The final answer is $\boxed{24}$. I hope it is correct.""",
    ),
    (
        # Example 3: Algebra / Word problem
        "Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?",
        r"""If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\cdot 12\cdot20=480$ pounds of weight. If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\cdot15\cdot n=30n$ pounds of weight. Equating this to 480 pounds, we can solve for $n$:
\begin{align*}
30n&=480\\
n&=\frac{480}{30}=\boxed{16}
\end{align*}
Final Answer: The final answer is $\boxed{16}$. I hope it is correct.""",
    ),
    (
        # Example 4: Algebra / System of equations
        r"If the system of equations \begin{align*} 6x-4y&=a,\\ 6y-9x &=b. \end{align*} has a solution $(x, y)$ where $x$ and $y$ are both nonzero, find $\frac{a}{b},$ assuming $b$ is nonzero.",
        r"""If we multiply the first equation by $-\frac{3}{2}$, we obtain
$$6y-9x=-\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have
$$-\frac{3}{2}a=b\Rightarrow\frac{a}{b}=\boxed{-\frac{2}{3}}.$$
Final Answer: The final answer is $\boxed{-\frac{2}{3}}$. I hope it is correct.""",
    ),
]
