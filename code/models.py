"""
Causal-BERT Model Implementation
基于因果干预的BERT安全检测模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel, BertConfig
from transformers import DebertaPreTrainedModel, DebertaModel, DebertaConfig
from typing import Dict, Tuple, Optional


class CausalBERT(BertPreTrainedModel):
    """
    Causal-BERT: 基于反事实对比学习的因果干预BERT模型

    该模型通过反事实对比学习损失，强制模型学习实体-关系的因果效应，
    而不是依赖统计相关性。
    """

    def __init__(self, config: BertConfig, num_labels: int = 2,
                 dropout: float = 0.1, margin: float = 1.0):
        """
        Args:
            config: BERT配置
            num_labels: 分类标签数量
            dropout: Dropout比例
            margin: 对比学习的margin参数
        """
        super().__init__(config)
        self.num_labels = num_labels
        self.margin = margin

        # BERT编码器
        self.bert = BertModel(config)

        # 分类头
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

        # 用于对比学习的投影头
        self.projection = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(config.hidden_size, 128)  # 投影到低维空间
        )

        self.init_weights()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        cf_input_ids: Optional[torch.Tensor] = None,
        cf_attention_mask: Optional[torch.Tensor] = None,
        cf_token_type_ids: Optional[torch.Tensor] = None,
        cf_labels: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            input_ids: 输入token IDs [batch_size, seq_len]
            attention_mask: 注意力mask [batch_size, seq_len]
            token_type_ids: token type IDs [batch_size, seq_len]
            labels: 分类标签 [batch_size]
            cf_input_ids: 反事实样本的input_ids [batch_size, seq_len]
            cf_attention_mask: 反事实样本的attention_mask [batch_size, seq_len]
            cf_token_type_ids: 反事实样本的token_type_ids [batch_size, seq_len]
            cf_labels: 反事实样本的标签 [batch_size]
            return_embeddings: 是否返回嵌入向量

        Returns:
            包含loss、logits、embeddings等内容的字典
        """
        # 编码原始样本
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        pooled_output = outputs.pooler_output  # [batch_size, hidden_size]
        pooled_output = self.dropout(pooled_output)

        # 分类logits
        logits = self.classifier(pooled_output)  # [batch_size, num_labels]

        # 对比学习的投影
        embeddings = self.projection(pooled_output)  # [batch_size, projection_dim]
        embeddings = F.normalize(embeddings, p=2, dim=1)  # L2归一化

        result = {
            "logits": logits,
            "embeddings": embeddings,
            "hidden_states": pooled_output if return_embeddings else None,
        }

        # 计算损失
        loss = None
        if labels is not None:
            # 标准分类损失
            ce_loss = F.cross_entropy(logits, labels)

            # 如果提供了反事实样本，计算因果对比损失
            causal_loss = torch.tensor(0.0, device=logits.device)
            if cf_input_ids is not None:
                causal_loss = self._compute_causal_loss(
                    input_ids, attention_mask, token_type_ids,
                    cf_input_ids, cf_attention_mask, cf_token_type_ids,
                    labels, cf_labels
                )

            # 总损失 = 分类损失 + 因果对比损失
            loss = ce_loss + causal_loss

            result["ce_loss"] = ce_loss
            result["causal_loss"] = causal_loss

        result["loss"] = loss
        return result

    def _compute_causal_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor],
        cf_input_ids: torch.Tensor,
        cf_attention_mask: torch.Tensor,
        cf_token_type_ids: Optional[torch.Tensor],
        labels: torch.Tensor,
        cf_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算反事实对比学习损失

        该损失确保：对于同一实体，当关系改变时，嵌入向量应该有足够的距离。
        这强制模型关注关系词而非实体词。

        Args:
            input_ids: 原始样本
            cf_input_ids: 反事实样本
            labels: 原始标签
            cf_labels: 反事实标签

        Returns:
            因果对比损失标量
        """
        batch_size = input_ids.size(0)

        # 编码反事实样本
        cf_outputs = self.bert(
            input_ids=cf_input_ids,
            attention_mask=cf_attention_mask,
            token_type_ids=cf_token_type_ids,
        )
        cf_pooled = cf_outputs.pooler_output
        cf_embeddings = self.projection(cf_pooled)
        cf_embeddings = F.normalize(cf_embeddings, p=2, dim=1)

        # 获取原始样本的嵌入
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled = outputs.pooler_output
        embeddings = self.projection(pooled)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # 计算L2距离
        # 对于不同标签的样本，应该有足够距离
        distance = torch.norm(embeddings - cf_embeddings, p=2, dim=1)  # [batch_size]

        # 创建mask：只对标签不同的样本施加margin约束
        label_diff = (labels != cf_labels).float()

        # Margin-based contrastive loss
        # 对于标签不同的样本，要求 distance >= margin
        # loss = max(0, margin - distance) * label_diff
        margin_loss = F.relu(self.margin - distance) * label_diff

        return margin_loss.mean()

    def predict(self, text: str, tokenizer, device: str = "cpu") -> Dict[str, any]:
        """
        对单个文本进行预测

        Args:
            text: 输入文本
            tokenizer: BERT tokenizer
            device: 设备

        Returns:
            预测结果字典
        """
        self.eval()
        with torch.no_grad():
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding="max_length"
            )

            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = self.forward(**inputs)
            logits = outputs["logits"][0]
            probs = F.softmax(logits, dim=-1)

            prediction = torch.argmax(probs).item()
            confidence = probs[prediction].item()

            return {
                "prediction": prediction,
                "confidence": confidence,
                "probabilities": probs.cpu().tolist()
            }

    def get_embeddings(self, text: str, tokenizer, device: str = "cpu") -> torch.Tensor:
        """
        获取文本的因果嵌入向量

        Args:
            text: 输入文本
            tokenizer: BERT tokenizer
            device: 设备

        Returns:
            归一化的嵌入向量
        """
        self.eval()
        with torch.no_grad():
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding="max_length"
            )

            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = self.forward(**inputs, return_embeddings=True)
            return outputs["embeddings"][0].cpu()


class BaselineBERT(BertPreTrainedModel):
    """
    基线BERT模型：仅使用标准交叉熵损失训练
    用于与CausalBERT进行对比
    """

    def __init__(self, config: BertConfig, num_labels: int = 2, dropout: float = 0.1):
        super().__init__(config)
        self.num_labels = num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        result = {"logits": logits}

        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            result["loss"] = loss

        return result


class AdversarialBERT(BertPreTrainedModel):
    """
    对抗训练BERT模型 (PGD-BERT基线)
    使用Projected Gradient Descent进行对抗训练
    """

    def __init__(self, config: BertConfig, num_labels: int = 2,
                 dropout: float = 0.1, perturb_epsilon: float = 0.01,
                 perturb_steps: int = 3, step_size: float = 0.003):
        super().__init__(config)
        self.num_labels = num_labels
        self.perturb_epsilon = perturb_epsilon
        self.perturb_steps = perturb_steps
        self.step_size = step_size

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        enable_adversarial: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播，支持对抗训练
        """
        # 获取embedding层
        embedding_output = self.bert.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        )

        if labels is not None and enable_adversarial and self.training:
            # 对抗训练：生成对抗扰动
            perturb = torch.zeros_like(embedding_output)
            perturb.requires_grad = True

            # PGD攻击
            for _ in range(self.perturb_steps):
                # 前向传播（带扰动）
                perturbed_embedding = embedding_output + perturb

                outputs = self.bert(
                    inputs_embeds=perturbed_embedding,
                    attention_mask=attention_mask,
                )
                pooled = outputs.pooler_output
                logits = self.classifier(self.dropout(pooled))

                # 计算损失
                loss = F.cross_entropy(logits, labels)

                # 计算梯度
                loss.backward()

                # 更新扰动
                perturb_data = perturb.grad.data
                perturb = perturb + self.step_size * perturb_data.sign()
                perturb = perturb.detach()
                perturb.requires_grad = True

                # 投影到epsilon球内
                perturb = torch.clamp(perturb, -self.perturb_epsilon, self.perturb_epsilon)

            # 使用最终扰动进行前向传播
            final_embedding = embedding_output + perturb.detach()
            outputs = self.bert(inputs_embeds=final_embedding, attention_mask=attention_mask)
        else:
            # 正常前向传播
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        result = {"logits": logits}

        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            result["loss"] = loss

        return result


class CausalDeBERTa(DebertaPreTrainedModel):
    """
    Causal-DeBERTa: 将反事实对比学习(CCL)应用到DeBERTa-v3架构

    该模型将Causal-BERT的因果干预方法应用到更强大的DeBERTa-v3基础模型上，
    用于验证CCL loss的模型无关性（model-agnostic）。

    关键改进：
    - 使用DeBERTa-v3的disentangled attention机制
    - 保持与CausalBERT相同的CCL loss
    - 用于与Causal-BERT和Vanilla DeBERTa进行公平对比
    """

    def __init__(self, config: DebertaConfig, num_labels: int = 2,
                 dropout: float = 0.1, margin: float = 1.0):
        """
        Args:
            config: DeBERTa配置
            num_labels: 分类标签数量
            dropout: Dropout比例
            margin: 对比学习的margin参数
        """
        super().__init__(config)
        self.num_labels = num_labels
        self.margin = margin

        # DeBERTa编码器
        self.deberta = DebertaModel(config)

        # 分类头
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

        # 用于对比学习的投影头
        self.projection = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(config.hidden_size, 128)  # 投影到低维空间
        )

        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        cf_input_ids: Optional[torch.Tensor] = None,
        cf_attention_mask: Optional[torch.Tensor] = None,
        cf_token_type_ids: Optional[torch.Tensor] = None,
        cf_labels: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            input_ids: 输入token IDs [batch_size, seq_len]
            attention_mask: 注意力mask [batch_size, seq_len]
            token_type_ids: token type IDs [batch_size, seq_len]
            labels: 分类标签 [batch_size]
            cf_input_ids: 反事实样本的input_ids [batch_size, seq_len]
            cf_attention_mask: 反事实样本的attention_mask [batch_size, seq_len]
            cf_token_type_ids: 反事实样本的token_type_ids [batch_size, seq_len]
            cf_labels: 反事实样本的标签 [batch_size]
            return_embeddings: 是否返回嵌入向量

        Returns:
            包含loss、logits、embeddings等内容的字典
        """
        # 编码原始样本
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # DeBERTa的池化输出 (使用[CLS] token)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)

        # 分类logits
        logits = self.classifier(pooled_output)

        # 对比学习的投影
        embeddings = self.projection(pooled_output)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        result = {
            "logits": logits,
            "embeddings": embeddings,
            "hidden_states": pooled_output if return_embeddings else None,
        }

        # 计算损失
        loss = None
        if labels is not None:
            # 标准分类损失
            ce_loss = F.cross_entropy(logits, labels)

            # 如果提供了反事实样本，计算因果对比损失
            causal_loss = torch.tensor(0.0, device=logits.device)
            if cf_input_ids is not None:
                causal_loss = self._compute_causal_loss(
                    input_ids, attention_mask, token_type_ids,
                    cf_input_ids, cf_attention_mask, cf_token_type_ids,
                    labels, cf_labels
                )

            # 总损失 = 分类损失 + 因果对比损失
            loss = ce_loss + causal_loss

            result["ce_loss"] = ce_loss
            result["causal_loss"] = causal_loss

        result["loss"] = loss
        return result

    def _compute_causal_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor],
        cf_input_ids: torch.Tensor,
        cf_attention_mask: torch.Tensor,
        cf_token_type_ids: Optional[torch.Tensor],
        labels: torch.Tensor,
        cf_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算反事实对比学习损失（与CausalBERT相同）

        该损失确保：对于同一实体，当关系改变时，嵌入向量应该有足够的距离。
        这强制模型关注关系而非实体。

        Args:
            input_ids: 原始样本
            cf_input_ids: 反事实样本
            labels: 原始标签
            cf_labels: 反事实标签

        Returns:
            因果对比损失标量
        """
        batch_size = input_ids.size(0)

        # 编码反事实样本
        cf_outputs = self.deberta(
            input_ids=cf_input_ids,
            attention_mask=cf_attention_mask,
            token_type_ids=cf_token_type_ids,
        )
        cf_pooled = cf_outputs.last_hidden_state[:, 0, :]
        cf_embeddings = self.projection(cf_pooled)
        cf_embeddings = F.normalize(cf_embeddings, p=2, dim=1)

        # 获取原始样本的嵌入
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled = outputs.last_hidden_state[:, 0, :]
        embeddings = self.projection(pooled)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # 计算L2距离
        distance = torch.norm(embeddings - cf_embeddings, p=2, dim=1)

        # 创建mask：只对标签不同的样本施加margin约束
        label_diff = (labels != cf_labels).float()

        # Margin-based contrastive loss
        margin_loss = F.relu(self.margin - distance) * label_diff

        return margin_loss.mean()


class VanillaDeBERTa(DebertaPreTrainedModel):
    """
    基线DeBERTa模型：仅使用标准交叉熵损失训练
    用于与CausalDeBERTa进行对比
    """

    def __init__(self, config: DebertaConfig, num_labels: int = 2, dropout: float = 0.1):
        super().__init__(config)
        self.num_labels = num_labels

        self.deberta = DebertaModel(config)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        result = {"logits": logits}

        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            result["loss"] = loss

        return result
