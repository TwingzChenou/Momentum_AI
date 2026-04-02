import os
import great_expectations as gx
import great_expectations.expectations as gxe

ROOT_DIR = "."

def create_suites():
    context = gx.get_context(project_root_dir=ROOT_DIR)

    # 1. BRONZE SUITE
    suite_name = "bronze_stock_prices_suite"
    try:
        context.suites.get(suite_name)
        print(f"Suite {suite_name} already exists.")
    except:
        suite = context.suites.add(gx.ExpectationSuite(name=suite_name))
        # GX 1.x expectations
        suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="AdjClose"))
        suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="Ticker"))
        print(f"Created {suite_name}.")

    # 2. SILVER SUITE
    suite_name = "silver_weekly_prices_suite"
    try:
        context.suites.get(suite_name)
        print(f"Suite {suite_name} already exists.")
    except:
        suite = context.suites.add(gx.ExpectationSuite(name=suite_name))
        suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="AdjClose"))
        suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="Date"))
        print(f"Created {suite_name}.")

    # 3. GOLD SUITE
    suite_name = "gold_momentum_features_suite"
    try:
        context.suites.get(suite_name)
        print(f"Suite {suite_name} already exists.")
    except:
        suite = context.suites.add(gx.ExpectationSuite(name=suite_name))
        suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="SMA_12"))
        suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="SMA_26"))
        print(f"Created {suite_name}.")

if __name__ == "__main__":
    create_suites()
