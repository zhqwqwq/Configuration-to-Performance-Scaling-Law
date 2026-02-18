import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class ScalingLawForecaster(nn.Module):
    def __init__(
        self,
        base_model_name: str = "HuggingFaceTB/SmolLM2-135M",
        init_from_pretrained: bool = True,
        force_fp32: bool = False,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(base_model_name)
        if force_fp32:
            self.config.torch_dtype = torch.float32
        if init_from_pretrained:
            if force_fp32:
                self.base = AutoModel.from_pretrained(
                    base_model_name,
                    config=self.config,
                    torch_dtype=torch.float32,
                )
            else:
                self.base = AutoModel.from_pretrained(base_model_name, config=self.config)
        else:
            self.base = AutoModel.from_config(self.config)

        hidden_size = self.config.hidden_size

        act_cls = nn.ReLU
        self.num_mlp = nn.Sequential(
            nn.Linear(1, hidden_size * 2),
            act_cls(),
            nn.Linear(hidden_size * 2, hidden_size)
        )

        self.head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        input_ids: torch.LongTensor,
        is_number_mask: torch.BoolTensor,
        number_values_filled: torch.FloatTensor,
        attention_mask: torch.BoolTensor = None
    ) -> torch.FloatTensor:
        """
        Args:
            input_ids:          (batch, seq_len)
            is_number_mask:     (batch, seq_len)    bool mask for numeric tokens
            number_values_filled:(batch, seq_len)    float values (0 for non-numeric)
            attention_mask:     (batch, seq_len)    optional
        Returns:
            logits: (batch, seq_len) scalar predictions per token
        """
        # Text embeddings
        input_ids[input_ids == 49152] = 0 
        text_emb = self.base.get_input_embeddings()(input_ids)

        # Numeric MLP embeddings
        flat_vals = number_values_filled.view(-1, 1)
        mlp_out = self.num_mlp(flat_vals)  
        mlp_out = mlp_out.view_as(text_emb) 

        mask = is_number_mask.unsqueeze(-1)
        inputs_embeds = torch.where(mask, mlp_out, text_emb)

        outputs = self.base(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True
        )
        hidden = outputs.last_hidden_state 

        # Final scalar head
        logits = self.head(hidden).squeeze(-1)  
        return logits

