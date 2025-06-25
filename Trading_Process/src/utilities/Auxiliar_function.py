import requests
import json
import os
import pandas as pd
import numpy as np
from datetime import date
from math import floor, ceil
from typing import List, Optional
import pandas_gbq
from google.cloud import bigquery
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# BigQuery client
client = bigquery.Client("hdx-data-platform")

# Asset lists
list_thematics = ["DEFI11", "WEB311", "META11", "FOMO11"]
list_nasdaq = ["HASH11", "SOLH11", "BITH11", "ETHE11", "XRPH11"]

### Auxiliar functions
def format_brl(value):
    """Format number to Brazilian currency format"""
    return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def create_email_sentences(df):
    # Dictionary to translate operation types
    side_translation = {
        'BUY': 'compra',
        'SELL': 'venda'
    }
    
    # List to store sentences
    sentences = []
    
    # Iterate through DataFrame rows
    for _, row in df.iterrows():
        # Get values from row
        asset = row['Asset']
        side = side_translation[row['Side'].upper()]
        spread_side = "abaixo" if row['Side'] == "SELL" else "acima"
        amount = format_brl(row['Amount'])  # Format the amount in BRL
        
        # Create sentence
        sentence = f"Favor iniciar uma {side} de {asset} até {amount}, até 30 bps {spread_side} do justo"
        sentences.append(sentence)
    
    return sentences

def create_primary_sentences(df: pd.DataFrame) -> list:
    import locale
    locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')  # for Linux/macOS
    # On Windows, use: locale.setlocale(locale.LC_ALL, 'Portuguese_Brazil.1252')

    primary_values = get_todays_primary_values()
    frases = []

    for _, row in df.iterrows():
        fund_name = row['fund_name']
        cnpj = row['fund_cnpj']
        asset = row['asset']
        side = row['side']
        units = row['primary_units']

        valor_row = primary_values.loc[primary_values['ETF'] == asset]
        if not valor_row.empty:
            valor_estimado = valor_row.iloc[0]['Estimated Opening Amount']
            # Calculate total value: unit price × number of units
            total_value = valor_estimado * units
            valor_formatado = f"R$ {total_value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        else:
            valor_formatado = 'R$ N/A'

        if side.upper() == 'BUY':
            frase = f"For {fund_name} (CNPJ: {cnpj}): Creation of {units} units of {asset} @ {valor_formatado}"
        elif side.upper() == 'SELL':
            frase = f"For {fund_name} (CNPJ: {cnpj}): Redemption of {units} units of {asset} @ {valor_formatado}"
        else:
            frase = f"{fund_name} (CNPJ: {cnpj}) - side '{side}' not recognized."

        frases.append(frase)

    return frases

def get_todays_primary_values():
    etfs = ['HASH11', 'BITH11', 'ETHE11', 'SOLH11', 'XRPH11', 'DEFI11', 'WEB311', 'META11', 'FOMO11']
    url = "https://api2.hashdex.io/marketdata/v2/inav/"

    results = []

    for etf in etfs:
        response = requests.get(url + etf)
        data = response.json()

        asset = data['info']['fundName']
        lot_units = data['info']['numberOfSharesPerCreationUnit']
        valorestimated_open = data['info']['estimatedNetAssetValuePerShareForNextDate']
        valorInav = data['inavPerShare']
        
        opening_amount = lot_units * valorestimated_open
        inav_amount = lot_units * valorInav

        results.append({
            'ETF': etf,
            'Lot Units': lot_units,
            'Estimated Opening iNAV': valorestimated_open,
            'iNAV per Share': valorInav,
            'Estimated Opening Amount': opening_amount,
            'Current Lot Amount': inav_amount
        })

    return pd.DataFrame(results)
        
def filter_primary_values(asset_trade):
    
    primary_values = get_todays_primary_values()
    primary_values = primary_values[primary_values['ETF'] == asset_trade]
    # print(primary_values)
    
    return primary_values
      
def calculate_primary_request(target_trade, lot_amount):
    
    # Calculate primary units request
    primary_units_request = ceil(target_trade / lot_amount) if (
        target_trade - (floor(target_trade / lot_amount)) * lot_amount > lot_amount / 2
    ) else floor(target_trade / lot_amount)
    
    return primary_units_request

def extract_asset_lists():
    """Extract unique onshore and offshore assets from funds configuration"""
    funds_data_static = pd.read_json(r'G:\Drives compartilhados\Investment Management\Trading_Process\src\Funds_stats_dict.json')
    # print(funds_data_static)
    onshore_assets = set()
    offshore_assets = set()
    
    for fund in funds_data_static["funds"]:
        # Add onshore assets (handle None values)
        onshore_list = fund.get("assets_onshore")
        if onshore_list is not None:
            onshore_assets.update(onshore_list)
        
        # Add offshore assets (handle None values)
        offshore_list = fund.get("assets_offshore")
        if offshore_list is not None:
            offshore_assets.update(offshore_list)
    
    return sorted(list(onshore_assets)), sorted(list(offshore_assets))

def get_and_organize_trades():
    
    trades_data = pd.read_csv(r'G:\Drives compartilhados\Investment Management\Trading_Process\src\Daily_trades_use.csv')
    # print(trades_data)
    simple_order, rebalance_order = trades_data[trades_data["Strategy"] == "SimpleOrder"], trades_data[trades_data["Strategy"] == "RebalanceOrder"]
    # print(simple_order)
    
    df_grouped_simple_order = simple_order.groupby(['Asset', 'Side'], as_index=False)['Amount'].sum()
    df_grouped_rebalance_order = rebalance_order.groupby(['Asset', 'Side'], as_index=False)['Amount'].sum()
    
    # Extract asset lists to separate onshore and offshore
    on_shore_assets, off_shore_assets = extract_asset_lists()
    
    # Filter grouped orders to only include ONSHORE assets for sentence generation
    df_grouped_simple_order_onshore = df_grouped_simple_order[
        df_grouped_simple_order['Asset'].isin(on_shore_assets)
    ]
    
    # Create sentences ONLY for onshore assets
    sentences = create_email_sentences(df_grouped_simple_order_onshore)
    # print(sentences)

    # Filter trades by onshore and offshore assets
    off_shore_trades = trades_data[trades_data["Asset"].isin(off_shore_assets)]
    on_shore_trades = trades_data[trades_data["Asset"].isin(on_shore_assets)]
    
    return (simple_order, rebalance_order, sentences,
            df_grouped_simple_order, df_grouped_rebalance_order,  
            trades_data, on_shore_trades, off_shore_trades)

def group_orders_for_broker(allocation_df: pd.DataFrame) -> pd.DataFrame:
    """Group allocation results by Asset and Side for broker submission"""
    # Filter for non-zero allocations only
    orders_to_send = allocation_df[allocation_df['Qty_Allocation'] > 0].copy()
    
    if orders_to_send.empty:
        return pd.DataFrame()
    
    # Group by Asset and Side (similar to your existing grouping pattern)
    grouped = orders_to_send.groupby(['Asset', 'Side']).agg({
        'Qty_Allocation': 'sum',
        'Financial_Allocation': 'sum',
        'Price_Executed': 'mean'
    }).reset_index()
    
    # Rename to match your existing pattern
    grouped.columns = ['Asset', 'Side', 'Amount', 'Financial_Total', 'Avg_Price']
    
    return grouped

def create_adjustment_email_sentences(allocation_df: pd.DataFrame, executed_trades_df: pd.DataFrame, total_allocation_df: pd.DataFrame = None):
    """Create comprehensive email with tables showing targets vs executed trades"""
    
    if allocation_df.empty:
        return "No adjustment orders needed."

    # Get all possible ETFs from your lists
    all_etfs = list_nasdaq + list_thematics  # Combine your ETF lists
    
    # Create summary for buy and sell sides
    buy_summary = []
    sell_summary = []
    
    # Aggregate data by Asset and Side, including both MainOrder and AdjustmentOrder
    for etf in all_etfs:
        for side in ['BUY', 'SELL']:
            # Get all orders for this ETF and side (both MainOrder and AdjustmentOrder)
            etf_orders = allocation_df[
                (allocation_df['Asset'] == etf) & 
                (allocation_df['Side'] == side)
            ]
            
            # Start with sum of Financial_Allocation for this side
            if etf_orders.empty:
                target_financial = 0
            else:
                target_financial = etf_orders['Financial_Allocation'].sum()
            
            # Now add Adjustment_Needed values from ALL records for this ETF
            # But be careful to avoid double counting:
            # - Negative Adjustment_Needed: add absolute value to OPPOSITE side only
            # - Positive Adjustment_Needed: add to SAME side only
            all_etf_records = allocation_df[allocation_df['Asset'] == etf]
            
            for _, record in all_etf_records.iterrows():
                # Check if we have the new column names or old ones
                if 'Adjustment_Needed' in record.index:
                    adjustment_needed = record['Adjustment_Needed']
                elif 'Target_to_Adjustment' in record.index:
                    # Use Target_to_Adjustment directly - no sign flip needed
                    adjustment_needed = record['Target_to_Adjustment']
                else:
                    adjustment_needed = 0
                
                record_side = record['Side']
                
                if adjustment_needed > 0 and record_side == side:
                    # Positive adjustment: add to same side only
                    target_financial += adjustment_needed
                elif adjustment_needed < 0 and record_side != side:
                    # Negative adjustment: add absolute value to opposite side only
                    target_financial += abs(adjustment_needed)
                # Note: we don't add anything if:
                # - adjustment_needed > 0 and record_side != side (positive from other side)
                # - adjustment_needed < 0 and record_side == side (negative from same side)
            
            # Calculate executed from actual trades data (not from allocation DataFrame)
            side_code = 'C' if side == 'BUY' else 'V'
            etf_executed = executed_trades_df[
                (executed_trades_df['ATIVO'] == etf) & 
                (executed_trades_df['SIDE'] == side_code)
            ]
            
            if not etf_executed.empty:
                executed_financial = etf_executed['Financeiro'].sum()
            else:
                executed_financial = 0.0
            
            # Calculate difference
            diff = target_financial - executed_financial
            
            # Format financial values
            target_str = f"R$ {target_financial:,.2f}".replace(",", ".")
            executed_str = f"R$ {executed_financial:,.2f}".replace(",", ".")
            diff_str = f"R$ {diff:,.2f}".replace(",", ".")
            
            # Store in appropriate list
            row = f"{etf}\t{target_str}\t{executed_str}\t{diff_str}"
            if side == 'BUY':
                buy_summary.append(row)
            else:
                sell_summary.append(row)
    
    # Create email body
    email_body = "Trading Adjustment Instructions\n\n"
    email_body += "Obs: Favor verificar na tabela abaixo o que já foi executado e usar a coluna 'Diff target - exec' como parâmetro para as ordens\n"
    email_body += "Obs2: Favor não entrar com essas ordens no leilão, paramos no final do mercado\n\n"
    
    # Add buy table
    email_body += "ETF <compra>\tFinanceiro target\tFinanceiro exec.\tDiff target - exec.\n"
    email_body += "\n".join(buy_summary) + "\n\n"
    
    # Add sell table  
    email_body += "ETF <venda>\tFinanceiro target\tFinanceiro exec.\tDiff target - exec.\n"
    email_body += "\n".join(sell_summary) + "\n\n"
    
    # ===========================================
    # ✅ NEW AGGREGATED ORDER LOGIC - CORRECTED BY ETF
    # ===========================================
    
    # Use total_allocation_df if provided, otherwise use allocation_df
    df_for_aggregation = total_allocation_df if total_allocation_df is not None else allocation_df
    
    # Calculate aggregated adjustments needed BY ETF
    etf_adjustments = {}  # {ETF: {'sell': amount, 'buy': amount}}
    
    for _, record in df_for_aggregation.iterrows():
        asset = record['Asset']
        
        # Get adjustment value
        if 'Adjustment_Needed' in record.index:
            adjustment_needed = record['Adjustment_Needed']
        elif 'Target_to_Adjustment' in record.index:
            adjustment_needed = record['Target_to_Adjustment']
        else:
            adjustment_needed = 0
                
        # Initialize ETF entry if not exists
        if asset not in etf_adjustments:
            etf_adjustments[asset] = {'sell': 0, 'buy': 0}
            
        if adjustment_needed > 0:
            # Positive adjustment = more selling needed
            etf_adjustments[asset]['sell'] += adjustment_needed
        elif adjustment_needed < 0:
            # Negative adjustment = more buying needed (use absolute value)
            etf_adjustments[asset]['buy'] += abs(adjustment_needed)
    
    # Create aggregated order sentences by ETF
    specific_orders = []
    order_count = 1
    
    # Process each ETF separately
    for etf, adjustments in etf_adjustments.items():
        sell_needed = adjustments['sell']
        buy_needed = adjustments['buy']
        
        # Add sell order if needed
        if sell_needed > 1000:  # Only if significant amount
            sell_amount = int(round(sell_needed, -2))  # Round to nearest 100
            specific_orders.append(f"{order_count}. Favor iniciar uma venda de {etf} até R$ {sell_amount:,.2f}, até 30 bps abaixo do justo".replace(",", "."))
            order_count += 1
        
        # Add buy order if needed  
        if buy_needed > 1000:  # Only if significant amount
            buy_amount = int(round(buy_needed, -2))  # Round to nearest 100
            specific_orders.append(f"{order_count}. Favor iniciar uma compra de {etf} até R$ {buy_amount:,.2f}, até 30 bps acima do justo".replace(",", "."))
            order_count += 1
    
    # Add specific orders section
    if specific_orders:
        email_body += "=== ORDENS ESPECÍFICAS ===\n"
        email_body += "\n".join(specific_orders) + "\n"
    else:
        email_body += "=== ORDENS ESPECÍFICAS ===\nNenhuma ordem de ajuste necessária\n"
    
    return email_body

def create_final_allocation_for_broker(allocation_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create final consolidated allocation for broker (secondary + adjustment orders combined)
    
    Args:
        allocation_df: DataFrame from calculate_update_secondary_values()
        
    Returns:
        DataFrame with total quantities to execute per fund/asset/side
    """
    
    # Filter for all orders with quantities > 0 (both MainOrder and AdjustmentOrder)
    all_orders = allocation_df[allocation_df['Qty_Allocation'] > 0].copy()
    
    if all_orders.empty:
        print("No orders to allocate")
        return pd.DataFrame(columns=['trade_date', 'asset', 'side', 'asset_amount', 'exchange_code', 'fund', 'cnpj'])
    
    # Group by Fund + Asset + Side + CNPJ and sum quantities
    grouped = all_orders.groupby(['Fund', 'CNPJ', 'Asset', 'Side']).agg({
        'Qty_Allocation': 'sum'
    }).reset_index()
    
    # Get current date
    current_date = date.today().strftime("%Y-%m-%d")
    
    # Convert to integers (truncating)
    grouped['asset_amount_int'] = grouped['Qty_Allocation'].astype(int)
    
    # AJUSTE PARA GARANTIR QUE NENHUMA QTY FIQUE NÃO ALOCADA
    # Para cada Asset+Side, verificar se a soma bate com o total executado
    for (asset, side), group in grouped.groupby(['Asset', 'Side']):
        total_original = group['Qty_Allocation'].sum()
        total_int = group['asset_amount_int'].sum()
        difference = int(total_original) - total_int
        
        if difference > 0:
            # Adicionar a diferença ao fundo com maior alocação
            max_idx = group['asset_amount_int'].idxmax()
            grouped.loc[max_idx, 'asset_amount_int'] += difference
    
    # Create final allocation DataFrame in new format
    final_allocation = pd.DataFrame({
        'trade_date': current_date,
        'asset': grouped['Asset'],
        'side': grouped['Side'].map({'BUY': 'C', 'SELL': 'V'}),
        'asset_amount': grouped['asset_amount_int'],  # Usar a versão ajustada
        'exchange_code': 'XP (3)',  # Fixed broker code
        'fund': grouped['Fund'],
        'cnpj': grouped['CNPJ']
    })
    
    # Sort by Asset, then Side, then Fund for better organization
    final_allocation = final_allocation.sort_values(['asset', 'side', 'fund']).reset_index(drop=True)
    
    print(f"Created {len(final_allocation)} total allocation orders")
    
    return final_allocation

def save_lot_units(df: pd.DataFrame, assets_list: List[str]) -> pd.DataFrame:
    """
    Save lot units to BigQuery with user confirmation
    
    Args:
        df: DataFrame with allocation data (from solver)
        assets_list: List of assets to filter
        
    Returns:
        DataFrame with lot units and sentences
    """
    try:
        # Validate inputs
        if df.empty:
            raise ValueError("Input DataFrame is empty")
        if not assets_list:
            raise ValueError("Assets list is empty")
            
        # Filter and rename columns
        df = df[["Fund", "CNPJ", "Asset", "Side", "Qty_Primary"]]
        
        df_assets = df[(df["Asset"].isin(assets_list)) & (df["Qty_Primary"] != 0)].rename(columns={
            "Fund": "fund_name",
            "CNPJ": "fund_cnpj",
            "Asset": "asset",
            "Side": "side",
            "Qty_Primary": "primary_units"
        })

        if df_assets.empty:
            print("No records to save after filtering")
            return df_assets, []

        # Add request_date column
        current_date = date.today().strftime("%Y-%m-%d")
        df_assets["request_date"] = current_date
        list_sentences = create_primary_sentences(df_assets)
        # print(list_sentences)
        # Define BigQuery parameters from environment
        project_id = os.getenv('GOOGLE_CLOUD_PROJECT_ID_USERS', 'hdx-data-platform-users')
        table_id = "team_trading.primary_requests"
        
        # Initialize BigQuery client
        client = bigquery.Client(project=project_id)
        
        # Delete existing records for each fund_cnpj
        for fund_cnpj in df_assets['fund_cnpj'].unique():
            delete_query = f"""
            DELETE FROM `{project_id}.{table_id}`
            WHERE request_date = '{current_date}'
            AND fund_cnpj = '{fund_cnpj}'
            """
            # Execute delete query
            query_job = client.query(delete_query)
            query_job.result()  # Wait for query to complete
        
        # Ask user for confirmation before uploading to GCP
        print(f"\nReady to upload primary units to BigQuery (GCP):")
        for _, row in df_assets.iterrows():
            # Show more of the fund name - take first 3 words or up to 40 characters
            fund_words = row['fund_name'].split(' ')
            if len(fund_words) >= 3:
                fund_display = ' '.join(fund_words[:3])
            else:
                fund_display = row['fund_name']
            
            # Limit to 40 characters max for readability
            if len(fund_display) > 40:
                fund_display = fund_display[:37] + "..."
                
            side_pt = "Compra" if row['side'] == 'BUY' else "Venda"
            print(f"  - {fund_display}: {int(row['primary_units'])} units of {row['asset']} ({side_pt})")
        
        user_confirmation = input("\nDo you want to proceed with the upload? (Y/N): ").strip().upper()
        
        if user_confirmation == 'Y':
            # Upload to BigQuery
            pandas_gbq.to_gbq(
                df_assets,
                destination_table=table_id,
                project_id=project_id,
                if_exists="append",
                table_schema=[
                    {"name": "request_date", "type": "DATE"},
                    {"name": "fund_name", "type": "STRING"},
                    {"name": "fund_cnpj", "type": "STRING"},
                    {"name": "asset", "type": "STRING"},
                    {"name": "side", "type": "STRING"},
                    {"name": "primary_units", "type": "FLOAT"},
                ],
                api_method="load_csv"
            )
            print(f"Successfully saved {len(df_assets)} records to BigQuery")
        else:
            print("Upload to BigQuery cancelled by user")
        
        return df_assets, list_sentences
        
    except Exception as e:
        print(f"Error saving lot units to BigQuery: {str(e)}")
        raise

def save_secondary_allocation_trades(df: pd.DataFrame, 
                                   csv_folder_path: str = None) -> pd.DataFrame:
    """
    Save secondary allocation trades to BigQuery and CSV file
    
    Args:
        df: DataFrame with secondary allocation data (from create_final_allocation_for_broker)
        csv_folder_path: Path to save CSV file (optional, uses environment variable if not provided)
        
    Returns:
        DataFrame with processed and saved data
    """
    try:
        # Validate inputs
        if df.empty:
            raise ValueError("Input DataFrame is empty")
        
        # Get CSV folder path from environment if not provided
        if csv_folder_path is None:
            csv_folder_path = os.getenv('CSV_FOLDER_PATH', 
                                      r"G:\Drives compartilhados\Ops-Trading\Onshore\CSV B3 gestora - Executed Trades by fund")
            
        # Get current date
        current_date = date.today().strftime("%Y-%m-%d")
        
        # === SAVE TO BIGQUERY ===
        # Rename trade_date to request_date to match BigQuery table
        df_to_save = df.copy()
        if 'trade_date' in df_to_save.columns:
            df_to_save = df_to_save.rename(columns={'trade_date': 'request_date'})
        
        # Show what will be saved and ask for confirmation
        print(f"\nReady to upload secondary allocation trades to BigQuery (GCP):")
        print(f"Date: {current_date}")
        print(f"Total records: {len(df_to_save)}")
        print("\nSummary by Asset and Side:")
        summary = df.groupby(['asset', 'side']).agg({
            'asset_amount': 'sum'
        }).reset_index()
        for _, row in summary.iterrows():
            side_name = "Compra" if row['side'] == 'C' else "Venda"
            print(f"  - {row['asset']}: {int(row['asset_amount'])} units ({side_name})")
        
        print(f"\nDetailed allocation:")
        for _, row in df.iterrows():
            # Show more of the fund name - take first 3 words or up to 40 characters
            fund_words = row['fund'].split(' ')
            if len(fund_words) >= 3:
                fund_display = ' '.join(fund_words[:3])
            else:
                fund_display = row['fund']
            
            # Limit to 40 characters max for readability
            if len(fund_display) > 40:
                fund_display = fund_display[:37] + "..."
                
            side_name = "Compra" if row['side'] == 'C' else "Venda"
            print(f"  - {fund_display}: {int(row['asset_amount'])} units of {row['asset']} ({side_name})")
        
        user_confirmation = input(f"\nDo you want to proceed with the BigQuery upload? (Y/N): ").strip().upper()
        
        if user_confirmation == 'Y':
            # Define BigQuery parameters from environment
            project_id = os.getenv('GOOGLE_CLOUD_PROJECT_ID_USERS', 'hdx-data-platform-users')
            table_id = "team_trading.secondary_allocation_trades"
            
            # Initialize BigQuery client
            client = bigquery.Client(project=project_id)
            
            # Delete existing records for each fund on the same date
            for fund_name in df_to_save['fund'].unique():
                delete_query = f"""
                DELETE FROM `{project_id}.{table_id}`
                WHERE request_date = '{current_date}'
                AND fund = '{fund_name}'
                """
                # Execute delete query
                query_job = client.query(delete_query)
                query_job.result()  # Wait for query to complete
            
            # Upload to BigQuery
            pandas_gbq.to_gbq(
                df_to_save,
                destination_table=table_id,
                project_id=project_id,
                if_exists="append",
                table_schema=[
                    {"name": "request_date", "type": "DATE"},
                    {"name": "asset", "type": "STRING"},
                    {"name": "side", "type": "STRING"},
                    {"name": "asset_amount", "type": "BIGNUMERIC"},
                    {"name": "exchange_code", "type": "STRING"},
                    {"name": "fund", "type": "STRING"},
                    {"name": "cnpj", "type": "STRING"},
                ],
                api_method="load_csv"
            )
            
            print(f"Successfully saved {len(df_to_save)} records to BigQuery")
        else:
            print("Upload to BigQuery cancelled by user")
        
        # === SAVE TO CSV (always save CSV regardless of BigQuery confirmation) ===
        # Create filename
        filename = f"aloc_trading_{current_date}.csv"
        
        # Full file path
        full_path = os.path.join(csv_folder_path, filename)
        
        # Save original DataFrame to CSV (with trade_date column)
        df.to_csv(full_path, index=False)
        
        print(f"Successfully saved CSV to: {full_path}")
        
        return df_to_save
        
    except Exception as e:
        print(f"Error saving secondary allocation trades: {str(e)}")
        raise

def save_adjustment_orders_to_csv(adjustment_df: pd.DataFrame, 
                                 csv_path: str = None) -> pd.DataFrame:
    """
    Save adjustment orders to the Daily_trades_use.csv file
    
    Args:
        adjustment_df: DataFrame with adjustment allocation data
        csv_path: Path to the Daily_trades_use.csv file (uses environment variable if not provided)
        
    Returns:
        DataFrame with the saved adjustment orders
    """
    try:
        if adjustment_df.empty:
            print("No adjustment orders to save")
            return pd.DataFrame()
        
        # Get CSV path from environment if not provided
        if csv_path is None:
            csv_path = os.getenv('DAILY_TRADES_CSV_PATH', 
                               r"G:\Drives compartilhados\Investment Management\Trading_Process\src\Daily_trades_use.csv")
        
        # Read existing CSV
        existing_df = pd.read_csv(csv_path)
        
        # Remove any existing AdjustmentOrder entries for today
        existing_df = existing_df[existing_df['Strategy'] != 'AdjustmentOrder']
        
        # Prepare adjustment orders in the same format as Daily_trades_use.csv
        adjustment_orders = pd.DataFrame({
            'Fund': adjustment_df['Fund'],
            'CNPJ': adjustment_df['CNPJ'], 
            'Asset': adjustment_df['Asset'],
            'Side': adjustment_df['Side'],
            'Amount': adjustment_df['Target_to_Adjustment'].abs(),  # Use Target_to_Adjustment instead of Financial_Allocation
            'Strategy': 'AdjustmentOrder'
        })
        
        # Combine existing data with new adjustment orders
        combined_df = pd.concat([existing_df, adjustment_orders], ignore_index=True)
        
        # Save back to CSV
        combined_df.to_csv(csv_path, index=False)
        
        print(f"✅ Saved {len(adjustment_orders)} adjustment orders to {csv_path}")
        print("\nAdjustment Orders Summary:")
        for _, row in adjustment_orders.iterrows():
            side_name = "Compra" if row['Side'] == 'BUY' else "Venda"
            print(f"  - {row['Fund']}: {side_name} {row['Asset']} R$ {row['Amount']:,.2f}")
        
        return adjustment_orders
        
    except Exception as e:
        print(f"Error saving adjustment orders: {str(e)}")
        raise

