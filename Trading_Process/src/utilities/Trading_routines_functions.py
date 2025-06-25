import pandas as pd
from typing import Optional
from google.cloud import bigquery
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# BigQuery client with project ID from environment
project_id = os.getenv('GOOGLE_CLOUD_PROJECT_ID', 'hdx-data-platform')
client = bigquery.Client(project_id)

# Asset lists
list_thematics = ["DEFI11", "WEB311", "META11", "FOMO11"]
list_nasdaq = ["HASH11", "SOLH11", "BITH11", "ETHE11", "XRPH11"]

# Import functions from other modules
from .Post_trades_api import TradingAPIManager
from .Trading_allocation_calculation import TradeAllocationCalculator
from .Auxiliar_function import (
    get_and_organize_trades, 
    create_adjustment_email_sentences, 
    create_final_allocation_for_broker,
    save_lot_units
)

class TradingWorkflowManager:
    """Manager for the complete daily trading workflow"""
    
    def __init__(self, secrets_arn: str = 'hdx-routines-secret'):
        """
        Initialize the trading workflow manager
        
        Args:
            secrets_arn: AWS secrets ARN for trading API
        """
        self.secrets_arn = secrets_arn
        self.api_manager = None
        self.on_shore_trades = None
        self.off_shore_trades = None
        self.trades_data = None
        self._load_initial_data()
    
    def _load_initial_data(self):
        """Load initial trading data"""
        try:
            simple_order, rebalance_order, sentences, df_grouped_simple_order, df_grouped_rebalance_order, trades_data, on_shore_trades, off_shore_trades = get_and_organize_trades()
            self.on_shore_trades = on_shore_trades
            self.off_shore_trades = off_shore_trades
            self.trades_data = trades_data
        except Exception as e:
            print(f"Warning: Could not load initial data: {e}")
    
    def _get_api_manager(self):
        """Get or create API manager instance"""
        if self.api_manager is None:
            self.api_manager = TradingAPIManager(self.secrets_arn)
        return self.api_manager
    
    def run_initial_orders(self):
        """Run only the initial orders step"""
        return self.execute_daily_trading_workflow(step="initial")
    
    def run_primary_calculation(self, primary_type: str = None):
        """Run only the primary units calculation step with fresh data"""
        # ALWAYS reload fresh data for primary calculation
        print("🔄 Loading fresh trading data...")
        simple_order, rebalance_order, sentences, df_grouped_simple_order, df_grouped_rebalance_order, trades_data, on_shore_trades, off_shore_trades = get_and_organize_trades()
        
        # Update instance variables with fresh data
        self.on_shore_trades = on_shore_trades
        self.off_shore_trades = off_shore_trades
        self.trades_data = trades_data
        
        return self.execute_daily_trading_workflow(step="primary", primary_type=primary_type)
    
    def run_adjustment_cycle(self, max_iterations: int = 3):
        """Run only the adjustment cycles step"""
        return self.execute_daily_trading_workflow(step="adjustments", max_adjustment_iterations=max_iterations)
    
    def check_adjustment_status(self):
        """Quick check to see current adjustment status without running cycles"""
        if self.on_shore_trades is None:
            self._load_initial_data()
        
        trades_df = self._get_api_manager().get_consolidated_trades()
        calculator = TradeAllocationCalculator(trades_df, self.on_shore_trades)
        
        # Just get current allocation without running cycles
        allocation_results = calculator.calculate_update_secondary_values()
        email_body = create_adjustment_email_sentences(allocation_results, trades_df)
        
        print("Current adjustment status:")
        # print(email_body)
        
        return allocation_results
    
    def run_post_adjustment_analysis(self):
        """Run post-adjustment analysis to check final status"""
        if self.on_shore_trades is None:
            self._load_initial_data()
        
        trades_df = self._get_api_manager().get_consolidated_trades()
        calculator = TradeAllocationCalculator(trades_df, self.on_shore_trades)
        
        return calculator.run_post_adjustment_analysis()
    
    def complete_adjustment_workflow(self, max_iterations: int = 3):
        """Run complete adjustment workflow and return broker-ready orders"""
        
        # Run adjustment cycles
        print("Running adjustment cycles...")
        adjustment_results = self.run_adjustment_cycle(max_iterations)
        
        # Debug: Check what we got back
        # print(f"Type of adjustment_results: {type(adjustment_results)}")
        # print(f"Content: {adjustment_results}")
        
        if not adjustment_results:
            print("No adjustments were needed")
            return pd.DataFrame()
        
        # Handle different return types
        if isinstance(adjustment_results, list):
            # If it's a list, get the last iteration
            final_allocation = adjustment_results[-1]['full_allocation']
        elif isinstance(adjustment_results, dict):
            # If it's a dict, it might have 'adjustment_results' key
            if 'adjustment_results' in adjustment_results:
                iterations = adjustment_results['adjustment_results']
                # Handle empty iterations list - when no adjustments were needed
                if not iterations:
                    print("No adjustment iterations were completed - using current allocation")
                    # Get current allocation without adjustments
                    trades_df = self._get_api_manager().get_consolidated_trades()
                    trades_df= trades_df.loc[trades_df['CONTA_NOME'] == 'HASHDEX_GESTORA']
                    calculator = TradeAllocationCalculator(trades_df, self.on_shore_trades)
                    current_allocation = calculator.calculate_update_secondary_values()
                    final_allocation = current_allocation
                else:
                    final_allocation = iterations[-1]['full_allocation']
            else:
                # Or it might be the result directly
                final_allocation = adjustment_results.get('full_allocation')
                if final_allocation is None:
                    print("Could not find allocation data in results")
                    return pd.DataFrame()
        else:
            print(f"Unexpected result type: {type(adjustment_results)}")
            return pd.DataFrame()
        
        # Create broker orders
        print("\nCreating final broker orders...")
        broker_orders = create_final_allocation_for_broker(final_allocation)
        
        return {
            'adjustment_history': adjustment_results,
            'broker_orders': broker_orders
        }
    
    def allocate_actual_executed_trades(self):
        """
        Allocate ONLY actual executed trades using solver allocation
        
        Returns:
            dict: Results with broker orders for actual executed trades
        """
        from datetime import date
        from .Solver_Allocation import AllocationSolver
        from google.cloud import bigquery
        
        print("🎯 === ALLOCATING ACTUAL EXECUTED TRADES ===")
        
        try:
            # Ensure we have fresh target data
            if self.on_shore_trades is None:
                self._load_initial_data()
            
            # Get actual executed trades from API
            trades_df = self._get_api_manager().get_consolidated_trades()
            trades_df= trades_df.loc[trades_df['CONTA_NOME'] == 'HASHDEX_GESTORA']
            print("\n📊 Actual Executed Trades:")
            print(trades_df[['SIDE', 'ATIVO', 'QTY. AVAILABLE', 'AVG_PRICE', 'Financeiro']])
            
            if trades_df.empty:
                print("⚠️ No executed trades found")
                return {'status': 'no_trades', 'broker_orders': pd.DataFrame()}
            
            # 🚀 GET PRIMARY UNITS FROM GCP (CRITICAL FIX!)
            current_date = date.today().strftime("%Y-%m-%d")
            client = bigquery.Client(project="hdx-data-platform")
            
            query = f"""
            SELECT 
                fund_name, fund_cnpj, asset, side, primary_units, request_date
            FROM 
                `hdx-data-platform-users.team_trading.primary_requests`
            WHERE 
                request_date = '{current_date}'
            """
            
            df_primary_units = client.query(query).to_dataframe()
            print(f"Loaded {len(df_primary_units)} primary units records from GCP")
            
            # Process each executed trade individually using solver
            all_allocations = []
            
            for _, trade in trades_df.iterrows():
                asset = trade['ATIVO']
                side_code = trade['SIDE']  # 'C' or 'V'
                side = 'BUY' if side_code == 'C' else 'SELL'
                qty_executed = int(trade['QTY. AVAILABLE'])
                price_executed = float(trade['AVG_PRICE'])
                
                print(f"\n🔧 Allocating: {qty_executed} units of {asset} ({side}) @ R$ {price_executed:.2f}")
                
                # Get fund targets for this asset/side
                fund_targets = self.on_shore_trades[
                    (self.on_shore_trades['Asset'] == asset) & 
                    (self.on_shore_trades['Side'] == side)
                ].copy()
                
                if fund_targets.empty:
                    print(f"⚠️ No fund targets found for {asset} {side}")
                    continue
                    
                # Prepare solver input
                solver_input = fund_targets[["Fund", "Amount"]].rename(columns={
                    "Fund": "fund", 
                    "Amount": "value_target"
                })
                solver_input['cnpj'] = fund_targets['CNPJ'].values
                
                # Use AllocationSolver to distribute the ACTUAL executed quantity
                solver = AllocationSolver(
                    qty_executed=qty_executed,        # Actual executed quantity
                    price_executed=price_executed,    # Actual executed price
                    asset_trade=asset,
                    df_funds_target=solver_input,
                    max_iterations=1000
                )
                
                # 🚀 USE solve_with_fixed_primary INSTEAD OF solve!
                allocation_result = solver.solve_with_fixed_primary(df_primary_units)
                
                # Create allocation records - FIX INDEXING ISSUE
                allocation_result_reset = allocation_result.reset_index(drop=True)
                fund_targets_reset = fund_targets.reset_index(drop=True)
                
                for i, row in allocation_result_reset.iterrows():
                    if row['qty_allocation'] > 0:  # Only include non-zero allocations
                        allocation_record = {
                            'trade_date': date.today().strftime("%Y-%m-%d"),
                            'asset': asset,
                            'side': side_code,  # Use 'C'/'V' format
                            'asset_amount': int(row['qty_allocation']),
                            'exchange_code': 'XP (3)',
                            'fund': fund_targets_reset.iloc[i]['Fund'],
                            'cnpj': fund_targets_reset.iloc[i]['CNPJ']
                        }
                        all_allocations.append(allocation_record)
                
                allocated_qty = allocation_result['qty_allocation'].sum()
                print(f"✅ Allocated {allocated_qty} units across {len(allocation_result[allocation_result['qty_allocation'] > 0])} funds")
                
                # Verify we allocated exactly what was executed
                if allocated_qty != qty_executed:
                    print(f"⚠️ WARNING: Allocated {allocated_qty} but executed {qty_executed}")
            
            # Create final allocation DataFrame
            if all_allocations:
                # CONSOLIDATION FIX: Group by (fund, asset, side) to eliminate duplicates
                initial_df = pd.DataFrame(all_allocations)
                
                # Group by fund, asset, and side, then sum quantities
                consolidated = initial_df.groupby(['fund', 'cnpj', 'asset', 'side']).agg({
                    'asset_amount': 'sum',
                    'trade_date': 'first',  # Keep first trade date
                    'exchange_code': 'first'  # Keep first exchange code
                }).reset_index()
                
                # Reorder columns to match original format
                final_allocation_df = consolidated[['trade_date', 'asset', 'side', 'asset_amount', 'exchange_code', 'fund', 'cnpj']]
                
                print(f"\n📋 FINAL ALLOCATION SUMMARY (CONSOLIDATED):")
                print(f"Total allocation records after consolidation: {len(final_allocation_df)}")
                print(f"Records before consolidation: {len(initial_df)}")
                
                # Group by asset and side for summary
                summary = final_allocation_df.groupby(['asset', 'side']).agg({
                    'asset_amount': 'sum'
                }).reset_index()
                
                for _, row in summary.iterrows():
                    side_name = "Compra" if row['side'] == 'C' else "Venda"
                    print(f"  - {row['asset']}: {int(row['asset_amount'])} units ({side_name})")
                
                return {
                    'status': 'success',
                    'broker_orders': final_allocation_df,
                    'trades_processed': len(trades_df),
                    'allocations_created': len(final_allocation_df)
                }
            else:
                print("⚠️ No allocations created")
                return {'status': 'no_allocations', 'broker_orders': pd.DataFrame()}
                
        except Exception as e:
            print(f"❌ Error allocating actual trades: {e}")
            return {"status": "error", "message": str(e), 'broker_orders': pd.DataFrame()}
    
    def _filter_trades_by_fund_assets(self, trades_df, selected_assets):
        """Filter trades to only include funds that contain assets from the selected category"""
        import json
        
        # Load funds configuration
        funds_data_static = pd.read_json(r'G:\Drives compartilhados\Investment Management\Trading_Process\src\Funds_stats_dict.json')
        
        # Get CNPJs of funds that have any of the selected assets
        relevant_fund_cnpjs = set()
        
        for fund in funds_data_static["funds"]:
            fund_cnpj = fund.get("fund_cnpj")
            fund_name = fund.get("fund_name")
            onshore_assets = fund.get("assets_onshore") or []
            offshore_assets = fund.get("assets_offshore") or []
            
            # Check if fund has any of the selected assets (onshore or offshore)
            fund_assets = set(onshore_assets + offshore_assets)
            matching_assets = fund_assets.intersection(set(selected_assets))
            
            if matching_assets:
                relevant_fund_cnpjs.add(fund_cnpj)
        
        # Filter trades to only include relevant funds - STRIP WHITESPACE FROM CNPJ
        trades_df_clean = trades_df.copy()
        trades_df_clean['CNPJ'] = trades_df_clean['CNPJ'].str.strip()
        filtered_trades = trades_df_clean[trades_df_clean['CNPJ'].isin(relevant_fund_cnpjs)].copy()
        
        print(f"Filtered trades: {len(trades_df)} -> {len(filtered_trades)} records")
        print(f"Relevant funds: {len(relevant_fund_cnpjs)} funds have assets from selected category")
        
        return filtered_trades
    
    def execute_daily_trading_workflow(self, step: str = "all", max_adjustment_iterations: int = 3, primary_type: str = None):
        """
        Execute specific parts of the daily trading workflow
        
        Args:
            step: Which step to run - "initial", "primary", "adjustments", or "all"
            max_adjustment_iterations: Max adjustment cycles to run
            primary_type: Primary type for filtering trades
            
        Returns:
            dict: Results from the executed step(s)
        """
        
        results = {}
        
        if step in ["initial", "all"]:
            print("=== Step 1: Initial Orders ===")
            simple_order, rebalance_order, sentences, df_grouped_simple_order, df_grouped_rebalance_order, trades_data, on_shore_trades, off_shore_trades = get_and_organize_trades()
            
            # Update instance variables
            self.on_shore_trades = on_shore_trades
            self.off_shore_trades = off_shore_trades
            self.trades_data = trades_data
            
            results['initial_orders'] = {
                'sentences': sentences,
                'on_shore_trades': on_shore_trades,
                'off_shore_trades': off_shore_trades,
                'grouped_simple': df_grouped_simple_order,
                'grouped_rebalance': df_grouped_rebalance_order
            }
            
            print(f"Generated {len(sentences)} initial order sentences")
            # Here you would send the email
            # send_trading_email(sentences, ...)
            
            if step == "initial":
                return results
        
        if step in ["primary", "all"]:
            print("\n=== Step 2: Primary Units Calculation ===")
            
            # Ask user which primary type to calculate if not specified
            if primary_type is None:
                print("Which primary calculation do you want to run?")
                print("1. Thematics (DEFI11, WEB311, META11, FOMO11)")
                print("2. Nasdaq (HASH11, SOLH11, BITH11, ETHE11, XRPH11)")
                choice = input("Enter 1 for Thematics or 2 for Nasdaq: ")
                primary_type = "thematics" if choice == "1" else "nasdaq"
            
            # Select asset list based on choice
            if primary_type.lower() == "thematics":
                selected_assets = list_thematics
                print(f"Calculating primaries for THEMATICS: {selected_assets}")
            else:
                selected_assets = list_nasdaq
                print(f"Calculating primaries for NASDAQ: {selected_assets}")
            
            # For primary calculation, ALWAYS use the most current on_shore_trades
            # This ensures fresh data is used, especially when run_primary_calculation is called directly
            if 'initial_orders' in results:
                on_shore_trades = results['initial_orders']['on_shore_trades']
            else:
                # Use the instance variable which was updated with fresh data in run_primary_calculation
                on_shore_trades = self.on_shore_trades
                if on_shore_trades is None:
                    # Fallback: reload if somehow still None
                    _, _, _, _, _, _, on_shore_trades, _ = get_and_organize_trades()
                    trades_df= trades_df.loc[trades_df['CONTA_NOME'] == 'HASHDEX_GESTORA']
                    self.on_shore_trades = on_shore_trades
            
            print(f"Using on_shore_trades with {len(on_shore_trades)} total records")
            
            # Filter on_shore_trades to only include funds that have assets from selected category
            filtered_trades = self._filter_trades_by_fund_assets(on_shore_trades, selected_assets)
            
            trades_df = self._get_api_manager().get_consolidated_trades()
            result_df_for_primary_save = TradeAllocationCalculator(trades_df, filtered_trades).calculate_primary_units_w_executed_trades()
            saved_primary_data, primary_sentences = save_lot_units(result_df_for_primary_save, selected_assets)
            
            results['primary_data'] = {
                'saved_data': saved_primary_data,
                'sentences': primary_sentences,
                'allocation_df': result_df_for_primary_save
            }
            
            print("Primary units calculated and saved")
            
            if step == "primary":
                return results
        
        if step in ["adjustments", "all"]:
            print("\n=== Step 3: Adjustment Cycles ===")
            
            # Get on_shore_trades from previous step or use cached
            if 'initial_orders' in results:
                on_shore_trades = results['initial_orders']['on_shore_trades']
            elif self.on_shore_trades is not None:
                on_shore_trades = self.on_shore_trades
            else:
                # Reload if running this step independently
                _, _, _, _, _, _, on_shore_trades, _ = get_and_organize_trades()
                trades_df= trades_df.loc[trades_df['CONTA_NOME'] == 'HASHDEX_GESTORA']
                self.on_shore_trades = on_shore_trades
            
            trades_df = self._get_api_manager().get_consolidated_trades()
            calculator = TradeAllocationCalculator(trades_df, on_shore_trades)
            adjustment_results = calculator.run_adjustment_cycle(max_iterations=max_adjustment_iterations)
            
            results['adjustment_results'] = adjustment_results
            
            print(f"Completed {len(adjustment_results)} adjustment iterations")
        
        return results