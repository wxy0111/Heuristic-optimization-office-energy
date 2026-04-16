import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import jpholiday
import pulp
from sklearn.preprocessing import RobustScaler
import torch
import warnings
import os
import openai
from typing import Dict, Optional
import json
import time

warnings.filterwarnings('ignore')

from lstm_rl_model import DataProcessor, LSTMRLSystem, pmv_calculator



class AIRecommendationGenerator:
    """AI提案生成器，使用OpenAI ChatGPT-4o-mini生成节能建议"""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        """
        初始化AI提案生成器

        参数:
        api_key: OpenAI API密钥
        model: 使用的模型名称
        """
        openai.api_key = api_key
        self.model = model
        self.client = openai.OpenAI(api_key=api_key)

    def generate_energy_saving_proposal(self,
                                        current_indoor_temp: float,
                                        current_indoor_humidity: float,
                                        outdoor_temp: float,
                                        outdoor_humidity: float,
                                        optimal_temp: float,
                                        optimal_humidity: float,
                                        energy_savings_percent: float,
                                        ac_status: str = "オフ") -> str:
        """
        生成节能提案

        参数:
        current_indoor_temp: 当前室内温度
        current_indoor_humidity: 当前室内湿度
        outdoor_temp: 室外温度
        outdoor_humidity: 室外湿度
        optimal_temp: 推荐温度
        optimal_humidity: 推荐湿度
        energy_savings_percent: 节能百分比
        ac_status: 空调状态

        返回:
        生成的节能提案文本
        """

        # 构建提示词
        prompt = f"""以下の情報に基づいて、自然で丁寧かつ簡潔な省エネ提案文を生成してください：

現在の室内温度は{current_indoor_temp:.1f}℃、湿度は{current_indoor_humidity:.1f}%です。
室外温度は{outdoor_temp:.1f}℃、湿度は{outdoor_humidity:.1f}%です。
最適化モデルによる推奨温湿度設定は{optimal_temp:.1f}℃、{optimal_humidity:.1f}%です。
最適な温湿度設定を実現することで、快適性の向上と{energy_savings_percent:.1f}%の省エネが期待できます。

要求：
1. 自然で丁寧かつ簡潔な提案文を生成し、ユーザーにエアコンの設定方法や窓の開閉について案内してください
2. 季節感や室内外の温度差を考慮した実用的なアドバイスを含めてください
3. 快適性と省エネの両立を強調してください
4. 200文字以内で簡潔にまとめてください

提案文："""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": "あなたは省エネと快適性の専門家です。ユーザーに対して実用的で丁寧な温湿度管理のアドバイスを提供してください。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.7
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"AI提案生成エラー: {e}")
            # フォールバック提案
            return self._generate_fallback_proposal(current_indoor_temp, optimal_temp, energy_savings_percent)

    def _generate_fallback_proposal(self, current_temp: float, optimal_temp: float, savings: float) -> str:
        """APIエラー時のフォールバック提案"""
        if optimal_temp < current_temp:
            action = "エアコンで冷房設定を調整するか、窓を開けて自然換気"
        else:
            action = "エアコンで暖房設定を調整"

        return f"現在の{current_temp:.1f}℃から{optimal_temp:.1f}℃への調整により、{savings:.1f}%の省エネが期待できます。{action}をお試しください。"


class EnergyOptimizerWithAI:
    """AI提案生成機能付きエネルギー最適化器"""

    def __init__(self, trained_system, data_processor, original_data, openai_api_key: str):
        self.system = trained_system
        self.data_processor = data_processor
        self.original_data = original_data

        # AI提案生成器を初期化
        self.ai_generator = AIRecommendationGenerator(openai_api_key)

        # 添加历史温度跟踪
        self.previous_temp = None
        self.temp_history = []

        self._analyze_temperature_ranges()
        self._calculate_mean_energy()
        self.debug = True

        # AI提案履歴
        self.ai_proposals = []
        self._validate_data_scalers()
        self._validate_model_state()

    def _validate_model_state(self):
        """验证模型状态和数据一致性"""
        print("验证模型和数据状态...")

        # 验证数据标准化器
        indoor_temp_col = self.data_processor.col_mapping['indoor_temp']
        if indoor_temp_col in self.data_processor.feature_scalers:
            # 测试标准化
            test_temp = 23.0
            scaled = self._scale_single(indoor_temp_col, test_temp)
            recovered = self._inverse_single(indoor_temp_col, scaled)
            error = abs(recovered - test_temp)

            if error < 0.01:
                print("数据标准化验证通过")
            else:
                print(f"警告：数据标准化误差较大 {error:.6f}")

        # 验证预测一致性
        try:
            # 创建测试数据
            test_shape = (1, 6, len(self.original_data.columns) - 1)
            test_data = np.random.randn(*test_shape) * 0.5  # 标准化范围内

            # 两次预测应该相同
            pred1 = self.system.predict_with_lstm_only(test_data)
            pred2 = self.system.predict_with_lstm_only(test_data)

            diff = abs(pred1[0] - pred2[0])
            if diff < 1e-6:
                print("预测一致性验证通过")
            else:
                print(f"警告：预测不一致，差异: {diff}")

        except Exception as e:
            print(f"预测验证失败: {e}")

    def _validate_data_scalers(self):
        """验证数据标准化器是否正常工作"""
        try:
            indoor_temp_col = self.data_processor.col_mapping['indoor_temp']

            # 测试标准化和逆标准化
            test_temp = 23.0
            scaled = self._scale_single(indoor_temp_col, test_temp)
            recovered = self._inverse_single(indoor_temp_col, scaled)

            if abs(recovered - test_temp) > 0.1:
                print(f"警告：温度标准化可能有问题 {test_temp} -> {scaled} -> {recovered}")
            else:
                print("数据标准化验证通过")

        except Exception as e:
            print(f"数据标准化验证失败: {e}")


    # 新增
    # ====== 新增：辅助缩放函数 ======
    def _scale_single(self, feature_name: str, raw_value: float) -> float:
        """将原始值按对应特征的scaler标准化（若无scaler回退到z-score，再无则原样返回）"""
        scaler = self.data_processor.feature_scalers.get(feature_name)
        if scaler is not None:
            return float(scaler.transform([[raw_value]])[0][0])
        # 回退到 z-score
        if feature_name in self.original_data.columns:
            col = self.original_data[feature_name].dropna()
            if len(col) > 1:
                std = float(col.std())
                if std > 0:
                    return float((raw_value - float(col.mean())) / std)
        # 最后回退：原样返回
        return float(raw_value)

    def _inverse_single(self, feature_name: str, scaled_value: float) -> float:
        """将标准化值按对应特征的scaler逆变换为原始值（若无scaler则原样返回）"""
        scaler = self.data_processor.feature_scalers.get(feature_name)
        if scaler is not None:
            return float(scaler.inverse_transform([[scaled_value]])[0][0])
        return float(scaled_value)
#----------------------------------------------------------------------------------

    def is_working_hour(self, timestamp):
        if jpholiday.is_holiday(timestamp.date()):
            return False
        if timestamp.weekday() >= 5:
            return False
        hour = timestamp.hour
        return (9 <= hour <= 12) or (13 < hour <= 18)

    def _analyze_temperature_ranges(self):
        indoor_temp_col = self.data_processor.col_mapping['indoor_temp']

        if indoor_temp_col in self.original_data.columns:
            work_time_mask = self.original_data.index.map(self.is_working_hour)
            work_time_temps = self.original_data.loc[work_time_mask, indoor_temp_col].dropna()

            if len(work_time_temps) > 0:
                self.temp_min = max(work_time_temps.min() - 0.5, 18.0)
                self.temp_max = min(work_time_temps.max() + 0.5, 28.0)
                self.temp_mean = work_time_temps.mean()

                print(f"工作时间温度范围分析:")
                print(f"工作时间数据点数: {len(work_time_temps)}")
                print(f"原始温度范围: {work_time_temps.min():.1f}°C - {work_time_temps.max():.1f}°C")
                print(f"优化后最低温度: {self.temp_min:.1f}°C")
                print(f"优化后最高温度: {self.temp_max:.1f}°C")
                print(f"平均温度: {self.temp_mean:.1f}°C")
            else:
                print("警告：没有找到工作时间的温度数据，使用默认值")
                self.temp_min = 20.0
                self.temp_max = 26.0
                self.temp_mean = 23.0
        else:
            print("警告：未找到温度列，使用默认值")
            self.temp_min = 20.0
            self.temp_max = 26.0
            self.temp_mean = 23.0

    def _calculate_mean_energy(self):
        energy_col = self.data_processor.col_mapping['energy']
        if energy_col in self.original_data.columns:
            work_mask = self.original_data.index.map(self.is_working_hour)
            work_energy = self.original_data.loc[work_mask, energy_col].dropna()
            non_work_energy = self.original_data.loc[~work_mask, energy_col].dropna()

            self.mean_energy = self.original_data[energy_col].mean()
            self.work_mean_energy = work_energy.mean() if len(work_energy) > 0 else self.mean_energy
            self.non_work_mean_energy = non_work_energy.mean() if len(non_work_energy) > 0 else self.mean_energy
            self.max_energy = self.original_data[energy_col].quantile(0.95)
            self.min_energy = self.original_data[energy_col].quantile(0.05)

            print(f"能耗统计:")
            print(f"总体平均能耗: {self.mean_energy:.2f} Wh")
            print(f"工作时间平均能耗: {self.work_mean_energy:.2f} Wh")
            print(f"非工作时间平均能耗: {self.non_work_mean_energy:.2f} Wh")
        else:
            self.mean_energy = 500.0
            self.work_mean_energy = 600.0
            self.non_work_mean_energy = 400.0
            self.max_energy = 1000.0
            self.min_energy = 100.0

    def get_environmental_data(self, timestamp, sequence_data):
        """获取当前环境数据，正确处理缩放后的数据"""
        try:
            feature_names = self.original_data.columns.tolist()

            # 获取列映射
            indoor_temp_col = self.data_processor.col_mapping['indoor_temp']
            indoor_humidity_col = self.data_processor.col_mapping['indoor_humidity']
            outdoor_temp_col = self.data_processor.col_mapping['outdoor_temp']
            outdoor_humidity_col = self.data_processor.col_mapping['outdoor_humidity']

            # 获取索引位置
            indoor_temp_idx = feature_names.index(indoor_temp_col)
            indoor_humidity_idx = feature_names.index(indoor_humidity_col)

            # 从标准化数据中获取值
            scaled_indoor_temp = sequence_data[-1, indoor_temp_idx]
            scaled_indoor_humidity = sequence_data[-1, indoor_humidity_idx]

            # 逆向缩放获取真实温湿度值
            if indoor_temp_col in self.data_processor.feature_scalers:
                indoor_temp = \
                self.data_processor.feature_scalers[indoor_temp_col].inverse_transform([[scaled_indoor_temp]])[0][0]
            else:
                indoor_temp = scaled_indoor_temp

            if indoor_humidity_col in self.data_processor.feature_scalers:
                indoor_humidity = \
                self.data_processor.feature_scalers[indoor_humidity_col].inverse_transform([[scaled_indoor_humidity]])[
                    0][0]
            else:
                indoor_humidity = scaled_indoor_humidity

            # 处理室外温湿度
            outdoor_temp = indoor_temp + np.random.uniform(-3, 3)  # 默认估算值
            outdoor_humidity = indoor_humidity + np.random.uniform(-10, 10)  # 默认估算值

            # 如果有室外数据列，则使用真实数据
            if outdoor_temp_col in feature_names:
                outdoor_temp_idx = feature_names.index(outdoor_temp_col)
                scaled_outdoor_temp = sequence_data[-1, outdoor_temp_idx]
                if outdoor_temp_col in self.data_processor.feature_scalers:
                    outdoor_temp = \
                    self.data_processor.feature_scalers[outdoor_temp_col].inverse_transform([[scaled_outdoor_temp]])[0][
                        0]
                else:
                    outdoor_temp = scaled_outdoor_temp

            if outdoor_humidity_col in feature_names:
                outdoor_humidity_idx = feature_names.index(outdoor_humidity_col)
                scaled_outdoor_humidity = sequence_data[-1, outdoor_humidity_idx]
                if outdoor_humidity_col in self.data_processor.feature_scalers:
                    outdoor_humidity = self.data_processor.feature_scalers[outdoor_humidity_col].inverse_transform(
                        [[scaled_outdoor_humidity]])[0][0]
                else:
                    outdoor_humidity = scaled_outdoor_humidity

            # 数值合理性检查和修正
            indoor_temp = np.clip(indoor_temp, 15, 35)
            indoor_humidity = np.clip(indoor_humidity, 20, 90)
            outdoor_temp = np.clip(outdoor_temp, -10, 45)
            outdoor_humidity = np.clip(outdoor_humidity, 10, 100)

            # 调试输出（可选）
            if self.debug:
                print(f"环境数据获取成功:")
                print(f"  室内: {indoor_temp:.1f}°C, {indoor_humidity:.1f}%")
                print(f"  室外: {outdoor_temp:.1f}°C, {outdoor_humidity:.1f}%")

            return {
                'indoor_temp': float(indoor_temp),
                'indoor_humidity': float(indoor_humidity),
                'outdoor_temp': float(outdoor_temp),
                'outdoor_humidity': float(outdoor_humidity)
            }

        except Exception as e:
            print(f"环境数据获取错误: {e}")
            print(f"使用默认环境数据")
            # 返回合理的默认值
            return {
                'indoor_temp': 24.0,
                'indoor_humidity': 60.0,
                'outdoor_temp': 22.0,
                'outdoor_humidity': 55.0
            }

    def calculate_pmv_for_temp(self, base_row, target_temp, timestamp):
        """
        计算给定温度下的PMV值
        base_row: 标准化的序列数据行
        target_temp: 目标温度（原始物理值）
        """
        try:
            feature_names = self.original_data.columns.tolist()

            # 将base_row转为原始值
            raw_data = {}
            for feature_name in feature_names:
                if feature_name in base_row.index:
                    raw_value = self._inverse_single(feature_name, float(base_row[feature_name]))
                    raw_data[feature_name] = raw_value

            # 使用目标温度计算PMV
            return self._calculate_pmv_from_raw(raw_data, float(target_temp), timestamp)

        except Exception:
            return 0.0

    def predict_energy_for_temp(self, sequence_data, target_temp, time_index, timestamp):
        """
        修正版：确保数据转换一致性
        target_temp: 原始物理温度值
        """
        try:
            optimized_sequence = sequence_data.copy()
            feature_names = self.original_data.columns.tolist()

            indoor_temp_col = self.data_processor.col_mapping['indoor_temp']
            globe_temp_col = self.data_processor.col_mapping['globe_temp'] \
                if self.data_processor.col_mapping['globe_temp'] in feature_names else None

            indoor_temp_idx = feature_names.index(indoor_temp_col)
            globe_temp_idx = feature_names.index(globe_temp_col) if globe_temp_col else -1
            pmv_idx = feature_names.index('PMV') if 'PMV' in feature_names else -1

            # 1) 更新室内温度
            indoor_temp_scaled = self._scale_single(indoor_temp_col, float(target_temp))
            optimized_sequence[-1, indoor_temp_idx] = indoor_temp_scaled

            # 2) 同步更新黑球温度（保持原始温差）
            if globe_temp_idx != -1:
                # 获取当前的温差
                current_indoor_raw = self._inverse_single(indoor_temp_col, float(sequence_data[-1, indoor_temp_idx]))
                current_globe_raw = self._inverse_single(globe_temp_col, float(sequence_data[-1, globe_temp_idx]))
                temp_diff = current_globe_raw - current_indoor_raw

                # 计算新的黑球温度并标准化
                new_globe_raw = float(target_temp) + temp_diff
                globe_scaled = self._scale_single(globe_temp_col, new_globe_raw)
                optimized_sequence[-1, globe_temp_idx] = globe_scaled

            # 3) 计算新的PMV值
            if pmv_idx != -1:
                # 将更新后的sequence转为原始值计算PMV
                updated_raw_data = self._convert_sequence_to_raw(optimized_sequence[-1], feature_names)
                pmv_raw = self._calculate_pmv_from_raw(updated_raw_data, float(target_temp), timestamp)

                # PMV标准化后写回
                pmv_scaled = self._scale_single('PMV', pmv_raw)
                optimized_sequence[-1, pmv_idx] = pmv_scaled

            # 4) LSTM预测
            optimized_sequence = optimized_sequence.reshape(1, optimized_sequence.shape[0], optimized_sequence.shape[1])
            prediction = self.system.predict_with_lstm_only(optimized_sequence)
            energy_prediction = self.data_processor.inverse_transform_target(prediction)[0][0]

            # 5) 合理性检查（保持原有逻辑）
            is_work_hour = self.is_working_hour(timestamp)
            expected_mean = self.work_mean_energy if is_work_hour else self.non_work_mean_energy

            if np.isnan(energy_prediction) or np.isinf(energy_prediction):
                energy_prediction = expected_mean
            elif energy_prediction < self.min_energy * 0.5:
                energy_prediction = expected_mean * 0.7
            elif energy_prediction > self.max_energy * 1.5:
                energy_prediction = self.max_energy * 1.2

            return max(energy_prediction, 0.0)

        except Exception as e:
            if self.debug:
                print(f"能耗预测错误 (温度={target_temp}°C): {e}")
            is_work_hour = self.is_working_hour(timestamp)
            return self.work_mean_energy if is_work_hour else self.non_work_mean_energy

    def _convert_sequence_to_raw(self, scaled_sequence_row, feature_names):
        """将sequence中的一行标准化数据转为原始物理量字典"""
        raw_data = {}
        for i, feature_name in enumerate(feature_names):
            raw_value = self._inverse_single(feature_name, float(scaled_sequence_row[i]))
            raw_data[feature_name] = raw_value
        return raw_data

    def _calculate_pmv_from_raw(self, raw_data, target_temp_raw, timestamp):
        """使用一致的原始物理量计算PMV"""
        try:
            indoor_humidity_col = self.data_processor.col_mapping['indoor_humidity']
            globe_temp_col = self.data_processor.col_mapping['globe_temp']

            Ta = float(target_temp_raw)
            humidity = float(raw_data.get(indoor_humidity_col, 50.0))

            # 使用更新后的黑球温度
            Tg = float(raw_data.get(globe_temp_col, Ta + 1.0))

            _, pmv_values = pmv_calculator(
                [timestamp],
                np.array([Ta]),
                np.array([Tg]),
                np.array([humidity])
            )

            pmv_val = float(pmv_values[0])
            if np.isnan(pmv_val) or np.isinf(pmv_val):
                pmv_val = 0.0

            # 限制在合理范围内
            pmv_val = float(np.clip(pmv_val, -3.0, 3.0))
            return pmv_val

        except Exception:
            return 0.0


    def balanced_optimization(self, sequence_data, current_temp, timestamp, time_index):
        """平衡能耗和舒适度的优化方法"""
        try:
            max_temp_change = 3
            if self.previous_temp is not None:
                temp_lower = max(self.previous_temp - max_temp_change, self.temp_min)
                temp_upper = min(self.previous_temp + max_temp_change, self.temp_max)
            else:
                temp_lower = max(current_temp - max_temp_change, self.temp_min)
                temp_upper = min(current_temp + max_temp_change, self.temp_max)

            temp_candidates = np.linspace(temp_lower, temp_upper, 15)
            best_temp = current_temp
            best_score = float('inf')

            feature_names = self.original_data.columns.tolist()
            base_row = pd.Series(sequence_data[-1], index=feature_names)

            is_work_hour = self.is_working_hour(timestamp)
            expected_energy = self.work_mean_energy if is_work_hour else self.non_work_mean_energy

            for temp in temp_candidates:
                energy = self.predict_energy_for_temp(sequence_data, temp, time_index, timestamp)
                pmv = self.calculate_pmv_for_temp(base_row, temp, timestamp)

                energy_score = (energy / expected_energy) * 100
                comfort_score = abs(pmv) * 50

                if self.previous_temp is not None:
                    stability_score = abs(temp - self.previous_temp) * 20
                else:
                    stability_score = abs(temp - current_temp) * 10

                if is_work_hour:
                    if abs(pmv) > 0.5:
                        comfort_score += 100

                total_score = energy_score + comfort_score + stability_score

                if total_score < best_score:
                    best_score = total_score
                    best_temp = temp

            self.previous_temp = best_temp
            if len(self.temp_history) >= 10:
                self.temp_history.pop(0)
            self.temp_history.append(best_temp)

            final_energy = self.predict_energy_for_temp(sequence_data, best_temp, time_index, timestamp)
            final_pmv = self.calculate_pmv_for_temp(base_row, best_temp, timestamp)

            return best_temp, final_energy, final_pmv

        except Exception as e:
            if self.debug:
                print(f"平衡优化失败 (时间点 {time_index}): {e}")
            return current_temp, self.mean_energy, 0.0

    def generate_ai_proposal(self, timestamp, sequence_data, optimal_temp, original_energy, optimized_energy):
        """生成AI节能提案，修复节能计算和数据获取问题"""
        try:
            # 获取环境数据
            env_data = self.get_environmental_data(timestamp, sequence_data)

            # 修复节能百分比计算 - 防止除零和异常值
            if original_energy > 0 and not np.isnan(original_energy) and not np.isnan(optimized_energy):
                energy_savings_percent = max(0, (original_energy - optimized_energy) / original_energy * 100)
                # 限制节能效果在合理范围内
                energy_savings_percent = min(energy_savings_percent, 50.0)  # 最大50%节能
            else:
                energy_savings_percent = 0.0

            # 调试输出
            if self.debug:
                print(f"\n=== 节能计算调试信息 ===")
                print(f"原始能耗: {original_energy:.2f} Wh")
                print(f"优化能耗: {optimized_energy:.2f} Wh")
                print(f"节能量: {original_energy - optimized_energy:.2f} Wh")
                print(f"节能比例: {energy_savings_percent:.2f}%")
                print(f"环境数据: {env_data}")
                print("========================")

            # 推荐湿度（基于温度和当前湿度智能调整）
            current_humidity = env_data['indoor_humidity']
            if optimal_temp < env_data['indoor_temp']:  # 降温时
                optimal_humidity = min(current_humidity + 3, 70)  # 略微增加湿度
            else:  # 升温时
                optimal_humidity = max(current_humidity - 3, 40)  # 略微降低湿度

            # 基于温度差判断空调状态
            temp_diff = abs(optimal_temp - env_data['indoor_temp'])
            if temp_diff > 2.0:
                ac_status = "オン"
            elif temp_diff > 0.5:
                ac_status = "弱運転"
            else:
                ac_status = "オフ"

            # 只有当节能效果显著时才生成提案
            if energy_savings_percent >= 2.0:  # 至少2%节能才值得建议
                proposal = self.ai_generator.generate_energy_saving_proposal(
                    current_indoor_temp=env_data['indoor_temp'],
                    current_indoor_humidity=env_data['indoor_humidity'],
                    outdoor_temp=env_data['outdoor_temp'],
                    outdoor_humidity=env_data['outdoor_humidity'],
                    optimal_temp=optimal_temp,
                    optimal_humidity=optimal_humidity,
                    energy_savings_percent=energy_savings_percent,
                    ac_status=ac_status
                )

                proposal_data = {
                    'timestamp': timestamp,
                    'current_conditions': env_data,
                    'recommendations': {
                        'temperature': optimal_temp,
                        'humidity': optimal_humidity,
                        'ac_status': ac_status
                    },
                    'energy_data': {
                        'original_energy': original_energy,
                        'optimized_energy': optimized_energy,
                        'energy_savings_wh': original_energy - optimized_energy,
                        'energy_savings_percent': energy_savings_percent
                    },
                    'ai_proposal': proposal
                }

                self.ai_proposals.append(proposal_data)

                if self.debug:
                    print(f"\n=== AI提案生成 ({timestamp.strftime('%H:%M')}) ===")
                    print(f"当前室内: {env_data['indoor_temp']:.1f}°C, {env_data['indoor_humidity']:.1f}%")
                    print(f"推荐设定: {optimal_temp:.1f}°C, {optimal_humidity:.1f}%")
                    print(f"节能效果: {energy_savings_percent:.1f}%")
                    print(f"AI提案: {proposal}")
                    print("=" * 50)

                return proposal_data
            else:
                if self.debug:
                    print(f"节能效果不显著({energy_savings_percent:.1f}%)，跳过AI提案生成")
                return None

        except Exception as e:
            print(f"AI提案生成错误: {e}")
            import traceback
            traceback.print_exc()
            return None

    def optimize_workday_schedule(self, test_data, seq_length=6, generate_ai_proposals=True):
        print("开始优化工作日工作时间的温度设定...")

        energy_col = self.data_processor.col_mapping['energy']
        indoor_temp_col = self.data_processor.col_mapping['indoor_temp']

        X_test, y_test, _, _, _, _ = self.data_processor.prepare_data(
            test_data, target_col=energy_col, seq_length=seq_length, test_split=0.0, val_split=0.0
        )
        test_timestamps = test_data.index[seq_length:]
        optimization_results = []
        print(f"总共需要处理 {len(X_test)} 个时间点")

        self.previous_temp = None
        self.temp_history = []
        self.ai_proposals = []

        for i in range(len(X_test)):
            timestamp = test_timestamps[i]
            current_sequence = X_test[i]
            actual_energy = self.data_processor.inverse_transform_target(y_test[i:i + 1])[0][0] if len(
                y_test) > i else 0

            current_temp = self.temp_mean
            if i < len(test_data) - seq_length:
                try:
                    current_temp_idx = test_data.index.get_loc(timestamp)
                    current_temp = test_data.iloc[current_temp_idx][indoor_temp_col]
                    if np.isnan(current_temp) or np.isinf(current_temp):
                        current_temp = self.temp_mean
                except:
                    current_temp = self.temp_mean

            try:
                original_pred = self.system.predict_with_lstm_only(
                    current_sequence.reshape(1, -1, current_sequence.shape[-1]))
                original_energy = self.data_processor.inverse_transform_target(original_pred)[0][0]
                original_energy = max(0, min(original_energy, self.max_energy * 1.5))
            except:
                is_work = self.is_working_hour(timestamp)
                original_energy = self.work_mean_energy if is_work else self.non_work_mean_energy

            is_work_hour = self.is_working_hour(timestamp)

            if is_work_hour:
                try:
                    optimal_temp, optimal_energy, optimal_pmv = self.balanced_optimization(
                        current_sequence, current_temp, timestamp, i
                    )
                    optimization_type = "优化"
                except Exception as e:
                    if self.debug:
                        print(f"优化失败: {e}")
                    optimal_temp = current_temp
                    optimal_energy = original_energy
                    optimal_pmv = self.calculate_pmv_for_temp(
                        pd.Series(current_sequence[-1]), current_temp, timestamp)
                    optimization_type = "失败"
            else:
                optimal_temp = current_temp
                optimal_energy = original_energy
                optimal_pmv = self.calculate_pmv_for_temp(
                    pd.Series(current_sequence[-1]), current_temp, timestamp)
                optimization_type = "保持"

            # 修复节能计算 - 不要人为限制合理的节能效果
            energy_savings = max(0, original_energy - optimal_energy)

            # 只有当节能过于异常时才进行调整（比如超过80%）
            if energy_savings > original_energy * 0.8:
                print(f"警告：异常高节能效果 ({energy_savings / original_energy * 100:.1f}%)，调整为合理范围")
                energy_savings = original_energy * 0.3
                optimal_energy = original_energy - energy_savings

            # 生成AI提案（每5个时间点生成一次，避免API调用过频繁）
            ai_proposal_data = None
            if generate_ai_proposals and is_work_hour and i % 5 == 0:
                ai_proposal_data = self.generate_ai_proposal(
                    timestamp, current_sequence, optimal_temp, original_energy, optimal_energy
                )
                time.sleep(0.5)  # API调用间隔

            result = {
                'timestamp': timestamp,
                'is_working_hour': is_work_hour,
                'optimization_type': optimization_type,
                'current_temp': current_temp,
                'optimal_temp': optimal_temp,
                'actual_energy': actual_energy,
                'original_predicted_energy': original_energy,
                'optimized_energy': optimal_energy,
                'optimal_pmv': optimal_pmv,
                'energy_savings': energy_savings,
                'temp_change': optimal_temp - current_temp,
                'ai_proposal': ai_proposal_data['ai_proposal'] if ai_proposal_data else None
            }
            optimization_results.append(result)

            if (i + 1) % 50 == 0:
                processed_pct = (i + 1) / len(X_test) * 100
                print(f"已处理 {i + 1}/{len(X_test)} 个时间点 ({processed_pct:.1f}%)")

        return pd.DataFrame(optimization_results)

    def save_ai_proposals(self, filename: str = None):
        """保存AI提案到文件"""
        if not filename:
            filename = f"ai_proposals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        try:
            # 转换为可序列化格式
            serializable_proposals = []
            for proposal in self.ai_proposals:
                serializable_proposal = proposal.copy()
                serializable_proposal['timestamp'] = proposal['timestamp'].isoformat()
                serializable_proposals.append(serializable_proposal)

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(serializable_proposals, f, ensure_ascii=False, indent=2)

            print(f"AI提案已保存到: {filename}")
            return filename
        except Exception as e:
            print(f"保存AI提案失败: {e}")
            return None

    def analyze_optimization_results(self, results_df):
        print("\n========== 优化结果分析 ==========")
        total_points = len(results_df)
        working_hours = results_df['is_working_hour'].sum()
        optimized_points = (results_df['optimization_type'] == '优化').sum()
        ai_proposals_count = results_df['ai_proposal'].notna().sum()

        original_total = results_df['original_predicted_energy'].sum()
        optimized_total = results_df['optimized_energy'].sum()
        total_savings = original_total - optimized_total
        savings_percentage = (total_savings / original_total) * 100 if original_total != 0 else 0

        print(f"\n基本统计:")
        print(f"总时间点数: {total_points}")
        print(f"工作时间点数: {working_hours}")
        print(f"成功优化点数: {optimized_points}")
        print(f"AI提案生成数: {ai_proposals_count}")

        print(f"\n能耗对比:")
        print(f"优化前总能耗: {original_total:.2f} Wh")
        print(f"优化后总能耗: {optimized_total:.2f} Wh")
        print(f"总节能量: {total_savings:.2f} Wh")
        print(f"节能比例: {savings_percentage:.2f}%")

        # 显示部分AI提案示例
        if ai_proposals_count > 0:
            print(f"\n=== AI提案示例 ===")
            ai_proposal_samples = results_df[results_df['ai_proposal'].notna()]['ai_proposal'].head(3)
            for i, proposal in enumerate(ai_proposal_samples, 1):
                print(f"{i}. {proposal}")
                print("-" * 50)

        return results_df


def main_optimization_with_ai():
    print("========== 基于机器学习和AI的能耗优化系统 ==========")

    # 添加模型状态验证
    def verify_model_consistency():
        """验证模型加载的一致性"""
        import pickle
        import os

        # 检查关键文件是否存在
        required_files = [
            'models/lstm_model.pth',
            'models/best_lstm_params.pkl',
            'models/data_scalers.pkl'
        ]

        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            print(f"警告：缺少关键文件: {missing_files}")
            return False

        print("所有模型文件检查通过")
        return True

    if not verify_model_consistency():
        print("模型文件不完整，可能影响预测准确性")
    """带AI提案生成的主优化函数"""
    print("========== 基于机器学习和AI的能耗优化系统 ==========")

    # OpenAI API密钥 - 请在这里设置您的API密钥

    OPENAI_API_KEY = ('YOUR_OPENAI_API_KEY'
                      'YOUR_OPENAI_API_KEY')

    if OPENAI_API_KEY == "your-openai-api-key-here":
        print("警告: 请设置您的OpenAI API密钥")
        return None

    try:
        data_processor = DataProcessor()
        df = data_processor.load_data('merged_data.csv')
        # 加载训练时的数据标准化器
        scaler_path = 'models/data_scalers.pkl'
        if os.path.exists(scaler_path):
            try:
                import pickle
                with open(scaler_path, 'rb') as f:
                    scaler_data = pickle.load(f)
                    data_processor.feature_scalers = scaler_data['feature_scalers']
                    data_processor.energy_scaler = scaler_data['energy_scaler']
                    # 确保列映射一致
                    if 'col_mapping' in scaler_data:
                        data_processor.col_mapping = scaler_data['col_mapping']
                print("成功加载训练时的数据标准化器")
            except Exception as e:
                print(f"加载数据标准化器失败: {e}")
        else:
            print("警告：未找到训练时的数据标准化器")

        try:
            df = data_processor.calculate_and_add_pmv(df)
        except Exception as e:
            print(f"PMV计算警告: {e}")
        df = data_processor.add_time_features(df)
        hourly_df = data_processor.resample_and_aggregate(df, freq='10min')

        total_days = (hourly_df.index.max() - hourly_df.index.min()).days + 1
        rl_train_end = hourly_df.index.min() + pd.Timedelta(days=int(total_days * 0.8))
        test_data = hourly_df[hourly_df.index > rl_train_end]

        print(f"测试数据范围: {test_data.index.min()} 到 {test_data.index.max()}")
        print(f"测试数据点数: {len(test_data)}")

        seq_length = 6
        energy_col = data_processor.col_mapping['energy']

        # 使用较小的样本进行测试（避免过多API调用）
        sample_data = test_data

        X_sample, _, _, _, _, _ = data_processor.prepare_data(
            sample_data, target_col=energy_col, seq_length=seq_length, test_split=0.0, val_split=0.0
        )
        input_shape = (seq_length, X_sample.shape[2])
        system = LSTMRLSystem(input_shape)

        try:
            system.lstm_model.load('models/lstm_model.pth')
            system.rl_agent.load('models/rl_actor.pth', 'models/rl_critic.pth')
            print("成功加载预训练模型")
        except:
            print("警告: 未找到预训练模型，将使用随机初始化的模型")

        # 创建带AI功能的优化器
        optimizer = EnergyOptimizerWithAI(system, data_processor, hourly_df, OPENAI_API_KEY)

        # 执行优化并生成AI提案
        print("\n开始优化并生成AI提案...")
        results_df = optimizer.optimize_workday_schedule(sample_data, seq_length, generate_ai_proposals=True)

        # 分析结果
        final_results = optimizer.analyze_optimization_results(results_df)

        # 保存AI提案
        proposals_file = optimizer.save_ai_proposals()

        # 保存优化结果
        safe_csv_name = f"optimization_results_with_ai.csv"
        final_results.to_csv(safe_csv_name, index=False, encoding='utf-8-sig')
        print(f"优化结果已保存到 {safe_csv_name}")

        # 显示AI提案统计
        print(f"\n=== AI提案生成统计 ===")
        print(f"总AI提案数: {len(optimizer.ai_proposals)}")
        if len(optimizer.ai_proposals) > 0:
            avg_savings = np.mean([p['energy_data']['energy_savings_percent'] for p in optimizer.ai_proposals])
            print(f"平均节能效果: {avg_savings:.1f}%")

        print("\n========== 优化完成 ==========")
        return final_results, optimizer.ai_proposals

    except Exception as e:
        print(f"优化过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    results, proposals = main_optimization_with_ai()
