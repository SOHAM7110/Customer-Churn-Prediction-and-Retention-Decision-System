"""Raw loading + cleaning helpers shared by every stage."""
import pathlib
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
XLSX = ROOT / "data/raw/online_retail_II.xlsx"
CACHE = ROOT / "data/interim/transactions.parquet"

def load_raw(force: bool = False) -> pd.DataFrame:
    """Line-level transactions from both sheets, cached as parquet (the xlsx is slow to read)."""
    if CACHE.exists() and not force:
        return pd.read_parquet(CACHE)
    sheets = pd.read_excel(XLSX, sheet_name=None)          # both year sheets
    df = pd.concat(sheets.values(), ignore_index=True)
    df = df.rename(columns={
        "Invoice": "invoice", "StockCode": "stock_code", "Description": "description",
        "Quantity": "quantity", "InvoiceDate": "invoice_date", "Price": "price",
        "Customer ID": "customer_id", "Country": "country"})
    n0 = len(df)
    df = df.drop_duplicates()                               # sheets overlap in Dec 2010
    print(f"Dropped {n0 - len(df):,} exact duplicate rows (sheet overlap + true dupes)")
    df["invoice"] = df["invoice"].astype(str)
    df["stock_code"] = df["stock_code"].astype(str)
    df["customer_id"] = df["customer_id"].astype("Int64")
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["is_cancel"] = df["invoice"].str.startswith("C")
    df["line_revenue"] = df["quantity"] * df["price"]
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE)
    return df

def build_orders(df: pd.DataFrame) -> pd.DataFrame:
    """One row per genuine purchase invoice for identified customers.
    Excludes: cancellations, non-positive qty/price, missing customer_id."""
    ok = (~df.is_cancel) & (df.quantity > 0) & (df.price > 0) & df.customer_id.notna()
    o = (df[ok].groupby(["customer_id", "invoice"], as_index=False)
               .agg(date=("invoice_date", "min"), revenue=("line_revenue", "sum"),
                    n_items=("quantity", "sum"), country=("country", "first")))
    o["customer_id"] = o["customer_id"].astype(int)
    return o.sort_values(["customer_id", "date"]).reset_index(drop=True)
