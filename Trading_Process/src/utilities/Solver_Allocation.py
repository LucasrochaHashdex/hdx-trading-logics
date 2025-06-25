import requests
import json
import time
from datetime import datetime, time as dt_time, date
import schedule
import sys
import os
import pandas as pd
import numpy as np
import decimal
from pyathena import connect
import math
from math import floor, ceil
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List, Optional
from google.cloud import bigquery
from scipy.optimize import minimize

# BigQuery client
client = bigquery.Client("hdx-data-platform")

# Asset lists
list_thematics = ["DEFI11", "WEB311", "META11", "FOMO11"]
list_nasdaq = ["HASH11", "SOLH11", "BITH11", "ETHE11", "XRPH11"]

# Import functions from other modules
from .Auxiliar_function import filter_primary_values, calculate_primary_request
from .Post_trades_api import TradingAPIManager



@dataclass
class AllocationResult:
    """Data class to hold allocation results"""
    allocations: np.ndarray
    error: float
    metrics: pd.DataFrame
def create_zero_result(df_funds_target: pd.DataFrame, lot_amount: float) -> pd.DataFrame:
    """Create a result DataFrame with zero allocations"""
    df_result = df_funds_target.copy()
    df_result["qty_allocation"] = 0
    df_result["qty_allocation_rounded"] = 0
    df_result["financial_allocation"] = 0
    df_result["target_to_primary"] = df_result["value_target"]
    df_result["qty_primary_units"] = df_result["value_target"].apply(
        lambda x: calculate_primary_request(x, lot_amount)
    )
    df_result["financial_primary_units"] = df_result["qty_primary_units"] * lot_amount
    df_result["Squared_error"] = 0
    return df_result


class AllocationSolver:
    """Solver for optimal share allocation with primary unit constraints"""
    
    def __init__(self, 
                 qty_executed: int,
                 price_executed: float,
                 asset_trade: str,
                 df_funds_target: pd.DataFrame,
                 max_iterations: int = 1000):
        """
        Initialize the allocation solver
        
        Args:
            qty_executed: Total quantity to allocate
            price_executed: Price per share
            asset_trade: Asset identifier
            df_funds_target: DataFrame with fund targets
            max_iterations: Maximum optimization iterations
        """
        self.qty_executed = qty_executed
        self.price_executed = price_executed
        self.asset_trade = asset_trade
        self.df_funds_target = df_funds_target.copy()
        self.max_iterations = max_iterations
        
        # Get primary unit info
        self.primary_units = self._get_primary_units()
        
        # Handle case where no primary units data is found
        if self.primary_units.empty:
            print(f"Warning: No primary units data found for {asset_trade}")
            self.lot_amount = 1.0  # Default fallback value
        else:
            self.lot_amount = self.primary_units["Current Lot Amount"].iloc[0]
        
        # Pre-calculate constants
        self.target_values = self.df_funds_target['value_target'].values
        self.n_funds = len(self.df_funds_target)
        self.max_possible_allocations = np.floor(
            self.target_values / self.price_executed
        ).astype(int)
        
    def _get_primary_units(self) -> pd.DataFrame:
        """Get primary unit information for the asset"""
        return filter_primary_values(self.asset_trade)
    
    def _calculate_metrics(self, allocations: np.ndarray) -> Tuple[float, pd.DataFrame]:
        """Calculate error and metrics for given allocations"""
        df_temp = self.df_funds_target.copy()
        
        # Calculate financial metrics
        df_temp["qty_allocation"] = allocations
        df_temp["financial_allocation"] = allocations * self.price_executed
        df_temp["target_to_primary"] = df_temp["value_target"] - df_temp["financial_allocation"]
        
        # Calculate primary units
        df_temp["qty_primary_units"] = df_temp["target_to_primary"].apply(
            lambda x: max(0, calculate_primary_request(x, self.lot_amount))
        )
        
        # Check for negative primary units
        if (df_temp["target_to_primary"] < 0).any():
            return float('inf'), df_temp
        
        # Calculate final metrics
        df_temp["financial_primary_units"] = df_temp["qty_primary_units"] * self.lot_amount
        df_temp["Squared_error"] = ((df_temp["value_target"] - (df_temp["financial_primary_units"] + df_temp["financial_allocation"]))
              / df_temp["value_target"]) ** 2
        
        return df_temp["Squared_error"].sum(), df_temp
    
    def _is_valid_allocation(self, allocations: np.ndarray) -> bool:
        """Check if allocation satisfies primary unit constraints"""
        financial_allocation = allocations * self.price_executed
        target_to_primary = self.target_values - financial_allocation
        return not (target_to_primary < 0).any()
    
    def _get_initial_allocation(self) -> np.ndarray:
        """Calculate initial allocation respecting constraints"""
        # Initial proportional allocation
        total_target = self.target_values.sum()
        allocations = np.minimum(
            self.max_possible_allocations,
            np.floor(self.target_values / total_target * self.qty_executed).astype(int)
        )
        
        # Distribute remaining shares
        remaining = self.qty_executed - allocations.sum()
        if remaining > 0:
            available = self.max_possible_allocations - allocations
            fractions = np.where(
                available > 0,
                (self.target_values / total_target * self.qty_executed) - allocations,
                0
            )
            
            while remaining > 0 and available.any():
                idx = np.argmax(fractions * (available > 0))
                if available[idx] <= 0:
                    fractions[idx] = 0
                    continue
                    
                test_alloc = allocations.copy()
                test_alloc[idx] += 1
                
                if self._is_valid_allocation(test_alloc):
                    allocations[idx] += 1
                    available[idx] -= 1
                    remaining -= 1
                else:
                    fractions[idx] = 0
                    
        return allocations
    
    def _optimize_allocation(self, initial_allocation: np.ndarray) -> Tuple[np.ndarray, float, pd.DataFrame]:
        """Optimize allocation using local search"""
        best_allocations = initial_allocation.copy()
        best_error, _ = self._calculate_metrics(best_allocations)
        
        shift_sizes = [1, 2, 5, 10]
        
        for _ in range(self.max_iterations):
            improved = False
            
            # Try all shift sizes
            for shift_size in shift_sizes:
                if improved:
                    break
                    
                # Try moving shares between funds
                for i in range(self.n_funds):
                    if improved:
                        break
                        
                    for j in range(self.n_funds):
                        if i == j or best_allocations[i] < shift_size:
                            continue
                            
                        test_allocations = best_allocations.copy()
                        test_allocations[i] -= shift_size
                        test_allocations[j] += shift_size
                        
                        if self._is_valid_allocation(test_allocations):
                            error, _ = self._calculate_metrics(test_allocations)
                            
                            if error < best_error:
                                best_allocations = test_allocations
                                best_error = error
                                improved = True
                                break
            
            if not improved:
                break
        
        _, final_metrics = self._calculate_metrics(best_allocations)
        return best_allocations, best_error, final_metrics
    
    def solve(self) -> pd.DataFrame:
        """
        Solve the allocation problem and return updated DataFrame
        
        Returns:
            DataFrame with optimal allocation and metrics
        """
        # Handle zero inputs
        if self.qty_executed <= 0 or self.price_executed <= 0:
            print("Zero or invalid inputs detected. Returning zero allocation.")
            return create_zero_result(self.df_funds_target, self.lot_amount)
        
        # Get initial allocation
        initial_allocation = self._get_initial_allocation()
        
        # Optimize allocation
        best_allocations, best_error, final_metrics = self._optimize_allocation(initial_allocation)
        
        # Update DataFrame with results
        self.df_funds_target["qty_allocation"] = best_allocations
        self.df_funds_target["qty_allocation_rounded"] = best_allocations
        self.df_funds_target["financial_allocation"] = final_metrics["financial_allocation"]
        self.df_funds_target["target_to_primary"] = final_metrics["target_to_primary"]
        self.df_funds_target["qty_primary_units"] = final_metrics["qty_primary_units"]
        self.df_funds_target["financial_primary_units"] = final_metrics["financial_primary_units"]
        self.df_funds_target["Squared_error"] = final_metrics["Squared_error"]
        
        # GARANTIR QUE 100% DAS QUANTIDADES SEJAM SEMPRE ALOCADAS
        total_allocated = best_allocations.sum()
        if total_allocated != self.qty_executed and self.qty_executed > 0:
            difference = self.qty_executed - total_allocated
            if difference > 0:
                # Adicionar a diferença ao fundo com maior alocação
                max_idx = np.argmax(best_allocations)
                best_allocations[max_idx] += difference
                
                # Atualizar o DataFrame com a correção
                self.df_funds_target["qty_allocation"] = best_allocations
                self.df_funds_target["qty_allocation_rounded"] = best_allocations
                # Recalcular métricas finais
                _, updated_metrics = self._calculate_metrics(best_allocations)
                self.df_funds_target["financial_allocation"] = updated_metrics["financial_allocation"]
                self.df_funds_target["target_to_primary"] = updated_metrics["target_to_primary"]
                self.df_funds_target["qty_primary_units"] = updated_metrics["qty_primary_units"]
                self.df_funds_target["financial_primary_units"] = updated_metrics["financial_primary_units"]
                self.df_funds_target["Squared_error"] = updated_metrics["Squared_error"]
        
        # Print summary
        # print(f"Final squared error: {best_error}")
        # print(f"Total shares executed: {best_allocations.sum()}")
        
        return self.df_funds_target

    def _gradient_optimize_secondary(self, remaining_targets: np.ndarray, primary_financial: np.ndarray) -> np.ndarray:
        """
        🚀 NOVA OTIMIZAÇÃO POR GRADIENTES 
        Substitui o algoritmo de busca local por otimização matemática
        """
        if self.qty_executed <= 0 or remaining_targets.sum() <= 0:
            return np.zeros(len(remaining_targets), dtype=int)
        
        # Função objetivo: minimizar erro quadrático
        def objective(x):
            total_financial = primary_financial + x * self.price_executed
            errors = (self.target_values - total_financial) / self.target_values
            return np.sum(errors ** 2)
        
        # Gradiente analítico da função objetivo  
        def gradient(x):
            total_financial = primary_financial + x * self.price_executed
            errors = (self.target_values - total_financial) / self.target_values
            grad = -2 * self.price_executed * errors / self.target_values
            return grad
        
        # Limites por fundo baseados nos targets restantes
        bounds = []
        for i in range(len(remaining_targets)):
            max_units = max(0, int(remaining_targets[i] / self.price_executed))
            bounds.append((0, max_units))
        
        # Inicialização proporcional aos targets restantes
        if remaining_targets.sum() > 0:
            x0 = remaining_targets / remaining_targets.sum() * self.qty_executed
            max_units = np.array([b[1] for b in bounds])
            x0 = np.minimum(x0, max_units)
        else:
            x0 = np.zeros(len(remaining_targets))
        
        # Ajustar para satisfazer restrição de quantidade total
        if x0.sum() > 0:
            x0 = x0 * self.qty_executed / x0.sum()
            max_units = np.array([b[1] for b in bounds])
            x0 = np.minimum(x0, max_units)
        
        # Restrição: soma das alocações = quantidade executada
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - self.qty_executed}
        ]
        
        try:
            # Otimização com gradientes analíticos
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                jac=gradient,
                bounds=bounds,
                constraints=constraints,
                options={
                    'maxiter': 1000,
                    'ftol': 1e-12,
                    'disp': False
                }
            )
            
            if result.success:
                # Arredondamento inteligente que preserva restrições
                x_optimal = self._intelligent_rounding(result.x, remaining_targets)
                return x_optimal
            else:
                # Fallback para inicialização se otimização falhar
                return self._intelligent_rounding(x0, remaining_targets)
                
        except Exception as e:
            print(f"Gradient optimization failed: {e}. Using fallback allocation.")
            # Fallback para alocação proporcional simples
            return self._simple_proportional_allocation(remaining_targets)
    
    def _intelligent_rounding(self, x_continuous: np.ndarray, remaining_targets: np.ndarray) -> np.ndarray:
        """Arredondamento que preserva restrições de quantidade total e limites por fundo"""
        x_rounded = np.round(x_continuous).astype(int)
        
        # Aplicar limites máximos por fundo
        max_units = np.floor(remaining_targets / self.price_executed).astype(int)
        x_rounded = np.minimum(x_rounded, max_units)
        
        total_diff = self.qty_executed - x_rounded.sum()
        
        if total_diff > 0:
            # Adicionar unidades nos fundos com maior fração perdida
            fractions = x_continuous - x_rounded
            indices = np.argsort(fractions)[::-1]
            
            for i in range(total_diff):
                idx = indices[i % len(indices)]
                if x_rounded[idx] < max_units[idx]:
                    x_rounded[idx] += 1
                    
        elif total_diff < 0:
            # Remover unidades dos fundos com menor fração perdida
            fractions = x_continuous - x_rounded
            indices = np.argsort(fractions)
            
            for i in range(-total_diff):
                idx = indices[i % len(indices)]
                if x_rounded[idx] > 0:
                    x_rounded[idx] -= 1
        
        return x_rounded
    
    def _simple_proportional_allocation(self, remaining_targets: np.ndarray) -> np.ndarray:
        """Alocação proporcional simples como fallback"""
        total_remaining = remaining_targets.sum()
        if total_remaining <= 0:
            return np.zeros(len(remaining_targets), dtype=int)
        
        max_allocations = np.floor(remaining_targets / self.price_executed).astype(int)
        allocations = np.minimum(
            max_allocations,
            np.floor(remaining_targets / total_remaining * self.qty_executed).astype(int)
        )
        
        # Distribuir quantidade restante
        remaining = self.qty_executed - allocations.sum()
        available = max_allocations - allocations
        
        for i in range(remaining):
            if available.sum() <= 0:
                break
            idx = np.argmax(available)
            if available[idx] > 0:
                allocations[idx] += 1
                available[idx] -= 1
        
        return allocations

    def solve_with_fixed_primary(self, df_primary_units: pd.DataFrame) -> pd.DataFrame:
        """
        Solve allocation with pre-calculated primary units
        
        Args:
            df_primary_units: DataFrame with columns [fund_name, fund_cnpj, asset, side, primary_units]
            
        Returns:
            DataFrame with optimal allocation and metrics using fixed primary units
        """
        # Handle zero inputs
        if self.qty_executed <= 0 or self.price_executed <= 0:
            print("Zero or invalid inputs detected. Returning zero allocation.")
            return create_zero_result(self.df_funds_target, self.lot_amount)
        
        # Filter primary units for current asset and create CNPJ mapping
        asset_primary = df_primary_units[df_primary_units['asset'] == self.asset_trade].copy()
        
        # Create CNPJ to primary units mapping
        fund_primary_map = {}
        if not asset_primary.empty:
            try:
                asset_primary['primary_units'] = pd.to_numeric(asset_primary['primary_units'], errors='coerce').fillna(0)
                asset_primary['fund_cnpj_clean'] = asset_primary['fund_cnpj'].astype(str).str.strip()
                fund_primary_map = dict(zip(asset_primary['fund_cnpj_clean'], asset_primary['primary_units']))
            except Exception as e:
                print(f"Error creating primary units mapping: {e}")
                fund_primary_map = {}
        
        # Initialize result DataFrame
        df_result = self.df_funds_target.copy()
        
        # Map primary units using CNPJ
        if 'cnpj' in df_result.columns:
            df_result['cnpj_clean'] = df_result['cnpj'].astype(str).str.strip()
            df_result["qty_primary_units"] = df_result['cnpj_clean'].map(fund_primary_map).fillna(0)
        else:
            df_result["qty_primary_units"] = 0
        
        # Ensure numeric types and handle multiplication error
        try:
            df_result["qty_primary_units"] = pd.to_numeric(df_result["qty_primary_units"], errors='coerce').fillna(0)
            lot_amount_numeric = float(self.lot_amount) if self.lot_amount else 0
            df_result["financial_primary_units"] = df_result["qty_primary_units"] * lot_amount_numeric
        except (ValueError, TypeError) as e:
            print(f"Error calculating financial primary units: {e}")
            df_result["qty_primary_units"] = 0
            df_result["financial_primary_units"] = 0
        
        # Calculate remaining target for secondary allocation
        df_result["remaining_target"] = df_result["value_target"] - df_result["financial_primary_units"]
        
        # Check for negative remaining targets
        if (df_result["remaining_target"] < 0).any():
            print("Warning: Some funds have primary allocation exceeding target value")
            df_result["remaining_target"] = df_result["remaining_target"].clip(lower=0)
        
        # Calculate max possible secondary allocations based on remaining targets
        remaining_targets = df_result["remaining_target"].values
        max_secondary_allocations = np.floor(remaining_targets / self.price_executed).astype(int)
        
        # 🚀 USE NEW GRADIENT OPTIMIZATION INSTEAD OF LOCAL SEARCH
        primary_financial = df_result["financial_primary_units"].values
        best_allocations = self._gradient_optimize_secondary(remaining_targets, primary_financial)
        
        # Calculate final error for logging
        financial_allocation = best_allocations * self.price_executed
        total_financial = financial_allocation + primary_financial
        squared_errors = ((df_result["value_target"].values - total_financial) / df_result["value_target"].values) ** 2
        best_error = squared_errors.sum()
        
        # GARANTIR QUE 100% DAS QUANTIDADES SEJAM SEMPRE ALOCADAS
        total_allocated = best_allocations.sum()
        if total_allocated != self.qty_executed and self.qty_executed > 0:
            difference = self.qty_executed - total_allocated
            if difference > 0:
                # Adicionar a diferença ao fundo com maior alocação
                max_idx = np.argmax(best_allocations)
                best_allocations[max_idx] += difference
                
                # Atualizar métricas após correção
                financial_allocation = best_allocations * self.price_executed
                total_financial = financial_allocation + primary_financial
                squared_errors = ((df_result["value_target"].values - total_financial) / df_result["value_target"].values) ** 2
                best_error = squared_errors.sum()

        # Update result DataFrame with final allocations
        df_result["qty_allocation"] = best_allocations
        df_result["qty_allocation_rounded"] = best_allocations
        df_result["financial_allocation"] = financial_allocation
        df_result["target_to_primary"] = df_result["remaining_target"] - df_result["financial_allocation"]
        
        # Calculate final squared error
        df_result["Squared_error"] = squared_errors
        
        # Print summary
        # print(f"Final squared error: {best_error}")
        # print(f"Total shares executed: {best_allocations.sum()}")
        # print(f"Total primary units used: {df_result['qty_primary_units'].sum()}")
        
        return df_result