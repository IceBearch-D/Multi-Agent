# tests/test_sandbox_basic.py
from mse.sandbox import Sandbox


def test_sandbox_can_run_code():
    """沙盒能执行代码并返回结果"""
    sandbox = Sandbox(timeout=10)

    result = sandbox.execute(
        files={
            "hello.py": 'print("hello from sandbox")',
        },

        test_command="python hello.py",
    )

    print(f"exit_code: {result.exit_code}")
    print(f"stdout: {result.stdout}")
    print(f"status: {result.status}")
    print(f"duration: {result.duration}s")

    sandbox.close()

    assert result.exit_code == 0
    assert "hello from sandbox" in result.stdout


def test_sandbox_catches_error():
    """沙盒能捕获代码错误"""
    sandbox = Sandbox(timeout=10)

    result = sandbox.execute(
        files={
            "bad.py": 'print(1 / 0)',
        },
        test_command="python bad.py",
    )

    print(f"status: {result.status}")
    print(f"error: {result.error_message}")

    sandbox.close()

    assert result.exit_code != 0
    assert result.status == "runtime_error"


def test_sandbox_runs_pytest():
    """沙盒能跑 pytest 测试"""
    sandbox = Sandbox(timeout=30, network_disabled=False)

    result = sandbox.execute(
        files={
            "solution.py": """
def add(a, b):
    return a + b
""",
            "test_solution.py": """
from solution import add

def test_add():
    assert add(1, 2) == 3

def test_add_negative():
    assert add(-1, 1) == 0
""",
        },
        requirements="pytest\n",
        test_command="python -m pytest test_solution.py -v --tb=short",
    )

    print(result.summary())

    sandbox.close()

    assert result.exit_code == 0
    assert result.status == "passed"


def test_sandbox_catches_wrong_answer():
    """代码能跑但结果错了"""
    sandbox = Sandbox(timeout=30, network_disabled=False)

    result = sandbox.execute(
        files={
            "solution.py": """
def two_sum(nums, target):
    return [0, 0]  # 故意写错
""",
            "test_solution.py": """
from solution import two_sum

def test_two_sum():
    assert two_sum([2, 7, 11, 15], 9) == [0, 1]
""",
        },
        requirements="pytest\n",
        test_command="python -m pytest test_solution.py -v --tb=short",
    )

    print(f"status: {result.status}")
    print(f"error: {result.error_message}")

    sandbox.close()

    assert result.exit_code != 0
    assert result.status == "wrong_answer"


# 在 main.py 最后加上这段

if __name__ == "__main__":
    print("=" * 50)
    print("Test 1: test_sandbox_can_run_code")
    print("=" * 50)
    test_sandbox_can_run_code()

    print("=" * 50)
    print("Test 2: test_sandbox_catches_error")
    print("=" * 50)
    test_sandbox_catches_error()

    print("=" * 50)
    print("Test 3: test_sandbox_runs_pytest")
    print("=" * 50)
    test_sandbox_runs_pytest()

    print("=" * 50)
    print("Test 4: test_sandbox_catches_wrong_answer")
    print("=" * 50)
    test_sandbox_catches_wrong_answer()

    print("\n✅ All tests completed!")
