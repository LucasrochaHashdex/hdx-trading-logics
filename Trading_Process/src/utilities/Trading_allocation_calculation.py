import pandas as pd
import numpy as np
from datetime import date
from typing import List, Optional
from google.cloud import bigquery
import os

# BigQuery client
client = bigquery.Client("hdx-data-platform")

# Asset lists
list_thematics = ["DEFI11", "WEB311", "META11", "FOMO11"]
list_nasdaq = ["HASH11", "SOLH11", "BITH11", "ETHE11", "XRPH11"]

# Import functions from other modules
from .Solver_Allocation import AllocationSolver
from .Auxiliar_function import create_adjustment_email_sentences, create_final_allocation_for_broker, save_secondary_allocation_trades, save_adjustment_orders_to_csv
from .Post_trades_api import TradingAPIManager

class TradeAllocationCalculator:
    """Calculator for trade allocations with primary units and adjustment orders"""
    
    def __init__(self, executed_trades: pd.DataFrame, df_funds_target: pd.DataFrame):
        """
        Initialize calculator with trade data
        
        Args:
            executed_trades: DataFrame with executed trades
            df_funds_target: DataFrame with fund targets
        """
        self.executed_trades = executed_trades
        self.df_funds_target = df_funds_target
        self.side_map = {'BUY': 'C', 'SELL': 'V'}
        
        # Load CNPJ mapping once during initialization
        self.account_to_cnpj_map = self._load_cnpj_mapping()
        
    def _load_cnpj_mapping(self) -> dict:
        """Load the mapping from fund CNPJs to fund information"""
        try:
            import json
            with open(r'G:\Drives compartilhados\Investment Management\Trading_Process\src\Funds_stats_dict.json', 'r') as f:
                funds_data = json.load(f)
            
            # Create CNPJ to fund mapping (since there are no account codes in the JSON)
            cnpj_mapping = {}
            for fund in funds_data["funds"]:
                fund_cnpj = fund.get("fund_cnpj", "")
                fund_name = fund.get("fund_name", "")
                
                if fund_cnpj and fund_name:
                    cnpj_mapping[fund_cnpj] = {
                        'fund_name': fund_name,
                        'fund_cnpj': fund_cnpj
                    }
            
            print(f"Loaded CNPJ mapping for {len(cnpj_mapping)} funds")
            return cnpj_mapping
            
        except Exception as e:
            print(f"Error loading CNPJ mapping: {e}")
            return {}
    
    def _add_cnpj_to_executed_trades(self, executed_trades: pd.DataFrame) -> pd.DataFrame:
        """Add CNPJ and Fund information to executed trades"""
        executed = executed_trades.copy()
        
        # FIXED: The executed trades should already have CNPJ column from the API
        # If not, we can't map them since we don't have account codes
        if 'CNPJ' not in executed.columns:
            print("Warning: Executed trades missing CNPJ column")
            executed['CNPJ'] = ''
            executed['Fund'] = ''
        else:
            # Clean CNPJ column and map to fund names using our mapping
            executed['CNPJ'] = executed['CNPJ'].astype(str).str.strip()
            executed['Fund'] = executed['CNPJ'].map(
            lambda x: self.account_to_cnpj_map.get(x, {}).get('fund_name', '')
        )
        
        # Log unmapped CNPJs
        unmapped = executed[executed['Fund'] == '']
        if not unmapped.empty:
            print(f"Warning: {len(unmapped)} executed trades could not be mapped to fund names")
            print("Unmapped CNPJs:", unmapped['CNPJ'].unique())
            
        return executed
        
    def _process_single_trade(self, asset: str, side: str) -> pd.DataFrame:
        """Process a single trade allocation and return allocation DataFrame"""
        # Get target and executed trades
        target_df = self.df_funds_target[(self.df_funds_target['Asset'] == asset) & 
                                    (self.df_funds_target['Side'] == side)]
        
        # If no targets for this asset/side combination, return empty DataFrame
        if target_df.empty:
            return pd.DataFrame()
        
        # Get executed trades if any
        executed = self.executed_trades[(self.executed_trades['SIDE'] == self.side_map[side]) & 
                                    (self.executed_trades['ATIVO'] == asset)]
        
        # Set execution metrics - use zeros if no executions
        if executed.empty:
            qty_executed = 0
            price_executed = 0
        else:
            qty_executed = executed['QTY. AVAILABLE'].sum()
            price_executed = executed['AVG_PRICE'].mean()
        
        # Prepare solver input
        funds_target = target_df[["Fund", "Amount"]].rename(columns={
            "Fund": "fund", 
            "Amount": "value_target"
        })
        
        # Create fund name and CNPJ mapping
        fund_names = target_df[["Fund"]].to_dict()['Fund']
        fund_cnpjs = target_df.set_index('Fund')['CNPJ'].to_dict()
        
        # Store original target amounts for later use
        original_targets = target_df.set_index('Fund')['Amount'].to_dict()
        
        # Run solver
        try:
            solver_result = AllocationSolver(
                qty_executed=qty_executed,
                price_executed=price_executed,
                asset_trade=asset,
                df_funds_target=funds_target,
                max_iterations=10000
            ).solve()
            
            # Calculate financial values
            financial_allocation = solver_result['financial_allocation']  # Qty_Allocation * Price_Executed
            financial_primary = solver_result['financial_primary_units']   # Qty_Primary * INAV price * lot_amount
            
            # Calculate total executed and adjustment needed
            total_executed = financial_allocation + financial_primary
            
            # Create result DataFrame
            result_df = pd.DataFrame({
                'Fund': solver_result.index.map(lambda x: fund_names.get(x) if fund_names.get(x) and pd.notna(fund_names.get(x)) else f"UNKNOWN_FUND_{x}"),
                'CNPJ': solver_result.index.map(lambda x: fund_cnpjs.get(fund_names.get(x, '')) if fund_names.get(x) and fund_names.get(x) in fund_cnpjs else ''),
                'Asset': asset,
                'Side': side,
                'Qty_Allocation': solver_result['qty_allocation'],
                'Qty_Primary': solver_result['qty_primary_units'],
                'Price_Executed': price_executed,
                'Financial_Allocation': financial_allocation,
                'Financial_Primary': financial_primary,
                'Target_Amount': solver_result.index.map(lambda x: original_targets.get(fund_names.get(x, x), 0)),
                'Total_Executed': total_executed,
                'Adjustment_Needed': solver_result.index.map(lambda x: original_targets.get(fund_names.get(x, x), 0)) - total_executed,  # (Target Amount - Total Executed)
                'Error': solver_result['Squared_error'],
                'Strategy': 'MainOrder'  # Mark as main order
            })
            
            # Order columns
            columns = ['Fund', 'CNPJ', 'Asset', 'Side', 'Qty_Allocation', 'Qty_Primary', 
                    'Price_Executed', 'Financial_Allocation', 'Financial_Primary',
                    'Target_Amount', 'Total_Executed', 'Adjustment_Needed', 'Error', 'Strategy']
            result_df = result_df[columns]
            
            return result_df
            
        except Exception as e:
            print(f"Error processing {asset} {side}: {str(e)}")
            return pd.DataFrame()
    
    def calculate_primary_units_w_executed_trades(self) -> pd.DataFrame:
        """Calculate all allocations and return single consolidated DataFrame"""
        all_results = []
        
        for asset in self.df_funds_target['Asset'].unique():
            # Process both buy and sell sides
            for side in ['BUY', 'SELL']:
                result = self._process_single_trade(asset, side)
                if not result.empty:
                    all_results.append(result)
        
        # Return empty DataFrame if no results
        if not all_results:
            return pd.DataFrame(columns=['Fund', 'CNPJ', 'Asset', 'Side', 'Qty_Allocation', 'Qty_Primary', 
                                       'Price_Executed', 'Financial_Allocation', 'Financial_Primary',
                                       'Target_Amount', 'Total_Executed', 'Adjustment_Needed', 'Error', 'Strategy'])
        
        # Combine all results and sort
        final_result = pd.concat(all_results, ignore_index=True).sort_values(
            by=['Asset', 'Side', 'Fund']
        )
        
        return final_result
    
    def get_summary(self) -> pd.DataFrame:
        """Get summary of allocations by Asset and Side"""
        result = self.calculate_primary_units_w_executed_trades()
        if result.empty:
            return pd.DataFrame()
            
        summary = result.groupby(['Asset', 'Side']).agg({
            'Qty_Allocation': 'sum',
            'Qty_Primary': 'sum',
            'Financial_Allocation': 'sum',
            'Financial_Primary': 'sum',
            'Target_Amount': 'sum',
            'Total_Executed': 'sum',
            'Adjustment_Needed': 'sum',
            'Error': 'mean'
        }).round(2)
        
        return summary

    def _get_primary_units_from_gcp(self) -> pd.DataFrame:
        """Get primary units from GCP database"""
        try:
            from google.cloud import bigquery
            from datetime import date
            
            client = bigquery.Client("hdx-data-platform")
            
            # Use the correct column name and date format
            today = date.today().strftime("%Y-%m-%d")
            query = f"""
            SELECT 
                fund_name,
                fund_cnpj, 
                asset,
                side,
                primary_units
            FROM `hdx-data-platform-users.team_trading.primary_requests`
            WHERE request_date = "{today}"
            """
            
            df = client.query(query).to_dataframe()
            print(f"Loaded {len(df)} primary units records from GCP")
            return df
            
        except Exception as e:
            print(f"Error loading primary units from GCP: {e}")
            print("Continuing with zero primary units for all funds")
            return pd.DataFrame()  # Return empty DataFrame on error

    def calculate_update_secondary_values(self) -> pd.DataFrame:
        """
        Calculate adjustment needs using the same logic as primary calculation 
        but with fixed primary units from GCP.
        
        This should work exactly like _process_single_trade but uses 
        AllocationSolver.solve_with_fixed_primary() instead of solve().
        """
        # Get primary units from GCP database for all assets
        primary_units_df = self._get_primary_units_from_gcp()
        
        if primary_units_df.empty:
            print("⚠️ No primary units found in GCP for today")
            return pd.DataFrame()
        
        all_results = []
        
        # Process each asset/side combination (same as primary calculation)
        # BUT filter out funds that should only be processed in their specific primary units side
        for asset in self.df_funds_target['Asset'].unique():
            for side in ['BUY', 'SELL']:
                result = self._process_adjustment_trade(asset, side, primary_units_df)
                if not result.empty:
                    all_results.append(result)
            
        # Return empty DataFrame if no results
        if not all_results:
            return pd.DataFrame(columns=['Fund', 'CNPJ', 'Asset', 'Side', 'Qty_Allocation', 'Qty_Primary', 
                                       'Price_Executed', 'Financial_Allocation', 'Financial_Primary',
                                       'Target_Value', 'Target_to_Adjustment', 'Error', 'Strategy'])
        
        # Combine all results and sort
        final_result = pd.concat(all_results, ignore_index=True).sort_values(
            by=['Asset', 'Side', 'Fund']
        )
        
        # CRITICAL FIX: Remove duplicate records for funds that have specific primary units
        # If a fund has primary units for a specific asset/side, keep only that record
        if not primary_units_df.empty:
            # Create a set of (fund_cnpj, asset) combinations that have primary units
            primary_combinations = set()
            for _, row in primary_units_df.iterrows():
                if pd.notna(row.get('fund_cnpj')) and pd.notna(row.get('asset')):
                    primary_combinations.add((str(row['fund_cnpj']).strip(), row['asset'], row['side']))
            
            # Filter out duplicate records
            filtered_results = []
            for _, row in final_result.iterrows():
                fund_cnpj = str(row['CNPJ']).strip()
                asset = row['Asset'] 
                side = row['Side']
                
                # Check if this fund has primary units for this asset
                fund_has_primary = any(combo[0] == fund_cnpj and combo[1] == asset for combo in primary_combinations)
                
                if fund_has_primary:
                    # Only keep if this is the correct side for primary units
                    correct_side = any(combo == (fund_cnpj, asset, side) for combo in primary_combinations)
                    if correct_side:
                        filtered_results.append(row)
                else:
                    # No primary units for this fund/asset - keep all sides
                    filtered_results.append(row)
            
            if filtered_results:
                final_result = pd.DataFrame(filtered_results).reset_index(drop=True)
            else:
                final_result = pd.DataFrame()
            
        # Filter only significant adjustments (>1000 BRL)
        if not final_result.empty:
            significant_adjustments = final_result[
                abs(final_result['Target_to_Adjustment']) > 1000
            ].copy()
        else:
            significant_adjustments = pd.DataFrame()
        
        if significant_adjustments.empty:
            print("✅ No significant adjustments needed (all < 1000 BRL)")
            return pd.DataFrame()
        
        print(f"Generated {len(significant_adjustments)} adjustment orders")
        return significant_adjustments
    
    def _process_adjustment_trade(self, asset: str, side: str, primary_units_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process adjustment trade using primary units from GCP.
        Keep it simple and ensure primary units are correctly mapped.
        """
        from .Solver_Allocation import AllocationSolver
        
        # Get target and executed trades
        target_df = self.df_funds_target[(self.df_funds_target['Asset'] == asset) & 
                                    (self.df_funds_target['Side'] == side)]
        
        if target_df.empty:
            return pd.DataFrame()
        
        # Remove duplicates if any
        if len(target_df) != len(target_df.drop_duplicates(['Fund', 'CNPJ'])):
            target_df = target_df.drop_duplicates(['Fund', 'CNPJ'], keep='first')
        
        # Get executed trades
        executed = self.executed_trades[(self.executed_trades['SIDE'] == self.side_map[side]) & 
                                    (self.executed_trades['ATIVO'] == asset)]
        
        # Set execution metrics
        if executed.empty:
            qty_executed = 0
            price_executed = 0
        else:
            qty_executed = executed['QTY. AVAILABLE'].sum()
            price_executed = executed['AVG_PRICE'].mean()
                
        # Prepare solver input
        funds_target = target_df[["Fund", "Amount"]].rename(columns={
            "Fund": "fund", 
            "Amount": "value_target"
        })
        
        # Add CNPJ column for primary units mapping (required by solve_with_fixed_primary)
        funds_target['cnpj'] = target_df['CNPJ'].values
        
        # Use the solver with fixed primary units
        try:
            solver_result = AllocationSolver(
                qty_executed=qty_executed,
                price_executed=price_executed,
                asset_trade=asset,
                df_funds_target=funds_target,
                max_iterations=10000
            ).solve_with_fixed_primary(primary_units_df)
                
            # Create result DataFrame with CORRECTED ADJUSTMENT LOGIC
            result_df = pd.DataFrame({
                'Fund': target_df['Fund'].values,
                'CNPJ': target_df['CNPJ'].values,
                'Asset': asset,
                'Side': side,
                'Qty_Allocation': solver_result['qty_allocation'].values,
                'Qty_Primary': solver_result['qty_primary_units'].values,
                'Price_Executed': price_executed,
                'Financial_Allocation': solver_result['financial_allocation'].values,
                'Financial_Primary': solver_result['financial_primary_units'].values,
                'Target_Value': solver_result['value_target'].values,
                'Target_to_Adjustment': solver_result['value_target'].values - 
                                      (solver_result['financial_allocation'].values + 
                                       solver_result['financial_primary_units'].values),
                'Error': solver_result['Squared_error'].values,
                'Strategy': 'AdjustmentOrder'
            })
            
            # CRITICAL FIX: Determine correct adjustment side based on primary unit direction and adjustment sign
            for i in range(len(result_df)):
                target_to_adj = result_df.loc[i, 'Target_to_Adjustment']
                primary_side = side  # Original primary unit side (BUY or SELL)
                
                if abs(target_to_adj) > 1000:  # Only for significant adjustments
                    if target_to_adj < 0:  # Negative adjustment
                        if primary_side == 'SELL':
                            # SELL primary + negative adjustment = over-sold, need to BUY
                            result_df.loc[i, 'Side'] = 'BUY'
                        else:  # primary_side == 'BUY'
                            # BUY primary + negative adjustment = over-bought, need to SELL
                            result_df.loc[i, 'Side'] = 'SELL'
                    else:  # Positive adjustment
                        if primary_side == 'SELL':
                            # SELL primary + positive adjustment = under-sold, need to SELL more
                            result_df.loc[i, 'Side'] = 'SELL'
                        else:  # primary_side == 'BUY'
                            # BUY primary + positive adjustment = under-bought, need to BUY more
                            result_df.loc[i, 'Side'] = 'BUY'
            
            # Order columns
            columns = ['Fund', 'CNPJ', 'Asset', 'Side', 'Qty_Allocation', 'Qty_Primary', 
                    'Price_Executed', 'Financial_Allocation', 'Financial_Primary',
                    'Target_Value', 'Target_to_Adjustment', 'Error', 'Strategy']
            return result_df[columns]
            
        except Exception as e:
            print(f"Error processing {asset} {side}: {str(e)}")
            return pd.DataFrame()
    
    def _generate_complete_allocation_summary(self) -> pd.DataFrame:
        """
        Generate complete allocation summary similar to result_nasdaq format.
        Shows allocation status for ALL funds, not just those needing adjustments.
        
        Returns:
            pd.DataFrame: Complete allocation summary with columns similar to result_nasdaq
        """
        try:
            # Get primary units from GCP
            primary_units_df = self._get_primary_units_from_gcp()
            
            all_results = []
            
            # Process each unique asset/side combination
            for asset in self.df_funds_target['Asset'].unique():
                for side in self.df_funds_target['Side'].unique():
                    
                    # Get target funds for this asset/side
                    target_df = self.df_funds_target[
                        (self.df_funds_target['Asset'] == asset) & 
                        (self.df_funds_target['Side'] == side)
                    ].copy()
                    
                    if target_df.empty:
                        continue
                    
                    # Remove duplicates if any
                    if len(target_df) != len(target_df.drop_duplicates(['Fund', 'CNPJ'])):
                        target_df = target_df.drop_duplicates(['Fund', 'CNPJ'], keep='first')
                    
                    # Get executed trades for this asset/side
                    executed = self.executed_trades[
                        (self.executed_trades['SIDE'] == self.side_map[side]) & 
                        (self.executed_trades['ATIVO'] == asset)
                    ]
                    
                    # Set execution metrics
                    if executed.empty:
                        qty_executed = 0
                        price_executed = 0
                    else:
                        qty_executed = executed['QTY. AVAILABLE'].sum()
                        price_executed = executed['AVG_PRICE'].mean()
                    
                    # Prepare solver input (same as _process_adjustment_trade)
                    funds_target = target_df[["Fund", "Amount"]].rename(columns={
                        "Fund": "fund", 
                        "Amount": "value_target"
                    })
                    
                    # Add CNPJ column for primary units mapping
                    funds_target['cnpj'] = target_df['CNPJ'].values
                    
                    # Use the solver with fixed primary units (same logic as adjustment)
                    try:
                        solver_result = AllocationSolver(
                            qty_executed=qty_executed,
                            price_executed=price_executed,
                            asset_trade=asset,
                            df_funds_target=funds_target,
                            max_iterations=10000
                        ).solve_with_fixed_primary(primary_units_df)
                        
                        # Create result records using solver results (CORRECTED CALCULATION)
                        for i, fund_row in target_df.iterrows():
                            fund_name = fund_row['Fund']
                            fund_cnpj = fund_row['CNPJ']
                            target_amount = fund_row['Amount']
                            
                            # Get values from solver result
                            if i - target_df.index[0] < len(solver_result):
                                idx = i - target_df.index[0]  # Adjust index for solver result
                                qty_allocation = int(solver_result['qty_allocation'].iloc[idx])
                                qty_primary = int(solver_result['qty_primary_units'].iloc[idx])
                                financial_allocation = float(solver_result['financial_allocation'].iloc[idx])
                                financial_primary = float(solver_result['financial_primary_units'].iloc[idx])  # ✅ CORRECTED
                                error = float(solver_result['Squared_error'].iloc[idx])
                            else:
                                # Fallback values
                                qty_allocation = 0
                                qty_primary = 0
                                financial_allocation = 0
                                financial_primary = 0
                                error = 0
                            
                            # Calculate totals
                            total_executed = financial_allocation + financial_primary
                            adjustment_needed = target_amount - total_executed
                            
                            # Create result record
                            result_record = {
                                'Fund': fund_name,
                                'CNPJ': fund_cnpj,
                                'Asset': asset,
                                'Side': side,
                                'Qty_Allocation': qty_allocation,
                                'Qty_Primary': qty_primary,
                                'Price_Executed': price_executed,
                                'Financial_Allocation': financial_allocation,
                                'Financial_Primary': financial_primary,  # ✅ Now using solver result
                                'Target_Amount': target_amount,
                                'Total_Executed': total_executed,
                                'Adjustment_Needed': adjustment_needed,
                                'Error': error,
                                'Strategy': 'AdjustmentOrder'
                            }
                            
                            all_results.append(result_record)
                            
                    except Exception as e:
                        print(f"Error using solver for {asset} {side}: {e}")
                        # Continue without this asset/side
                        continue
            
            if all_results:
                result_df = pd.DataFrame(all_results)
                
                # Order columns to match result_nasdaq format
                columns = ['Fund', 'CNPJ', 'Asset', 'Side', 'Qty_Allocation', 'Qty_Primary', 
                          'Price_Executed', 'Financial_Allocation', 'Financial_Primary',
                          'Target_Amount', 'Total_Executed', 'Adjustment_Needed', 'Error', 'Strategy']
                
                return result_df[columns]
            else:
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error generating complete allocation summary: {e}")
            return pd.DataFrame()
    
    def run_adjustment_cycle(self, max_iterations: int = 3):
        """
        Run the complete adjustment cycle with iterative refinement.
        
        This implements the user's workflow:
        1. Calculate what adjustments are needed based on current executions
        2. Generate adjustment orders
        3. After adjustments are executed, recalculate for next iteration
        4. Continue until no significant adjustments needed
        """
        all_iterations = []
        
        for iteration in range(1, max_iterations + 1):
            print(f"\n=== Adjustment Iteration {iteration} ===")
            
            # Get fresh executed trades data
            updated_trades = TradingAPIManager('hdx-routines-secret').get_consolidated_trades()
            updated_trades = updated_trades.loc[updated_trades['CONTA_NOME'] == 'HASHDEX_GESTORA']
            self.executed_trades = updated_trades
            
            # Calculate current adjustment needs
            allocation_results = self.calculate_update_secondary_values()
            
            # Generate complete allocation summary (similar to result_nasdaq format)
            total_allocation_df = self._generate_complete_allocation_summary()
            
            # Create email with adjustment instructions
            email_body = create_adjustment_email_sentences(allocation_results, updated_trades, total_allocation_df)
            
            # Check if any significant adjustments are needed
            has_orders = not allocation_results.empty and "Nenhuma ordem de ajuste necessária" not in email_body
            
            if not has_orders:
                print("✅ No more significant adjustments needed")
                break
            
            # Store iteration results
            iteration_data = {
                'iteration': iteration,
                'email_body': email_body,
                'full_allocation': allocation_results,
                'total_allocation_df': total_allocation_df,  # NEW: Complete allocation summary
                'executed_trades_snapshot': updated_trades.copy()
            }
            all_iterations.append(iteration_data)
            
            print(f"📊 Generated {len(allocation_results)} adjustment orders for iteration {iteration}")
            
            # Save adjustment orders to CSV
            save_adjustment_orders_to_csv(allocation_results)
            
        # If no iterations were run, still provide the complete allocation summary
        if not all_iterations:
            print("✅ No adjustment iterations needed")
            # Get current state for complete summary
            updated_trades = TradingAPIManager('hdx-routines-secret').get_consolidated_trades()
            updated_trades = updated_trades.loc[updated_trades['CONTA_NOME'] == 'HASHDEX_GESTORA']
            self.executed_trades = updated_trades
            
            total_allocation_df = self._generate_complete_allocation_summary()
            
            # Return summary even when no adjustments needed
            return [{
                'iteration': 0,
                'email_body': 'No adjustments needed',
                'full_allocation': pd.DataFrame(),
                'total_allocation_df': total_allocation_df,
                'executed_trades_snapshot': updated_trades.copy()
            }]
        
        return all_iterations
    
    def run_post_adjustment_analysis(self) -> dict:
        """
        Run analysis after adjustment executions to see final status.
        
        This method should be called after adjustment orders have been executed
        to see the final state and prepare any remaining orders.
        """
        print("\n=== Post-Adjustment Analysis ===")
        
        # Get the most recent executed trades
        final_trades = TradingAPIManager('hdx-routines-secret').get_consolidated_trades()
        self.executed_trades = final_trades
        
        # Calculate final adjustment needs
        final_adjustments = self.calculate_update_secondary_values()
        
        # Create summary of final state
        summary = {
            'final_adjustments': final_adjustments,
            'total_remaining_orders': len(final_adjustments),
            'executed_trades_final': final_trades,
            'needs_further_adjustment': not final_adjustments.empty
        }
        
        if final_adjustments.empty:
            print("✅ All targets achieved - no further adjustments needed")
            summary['status'] = 'COMPLETE'
        else:
            print(f"⚠️  {len(final_adjustments)} adjustments still needed")
            summary['status'] = 'PENDING'
            
            # Create final broker orders if needed
            broker_orders = create_final_allocation_for_broker(final_adjustments)
            summary['broker_orders'] = broker_orders
            
        return summary