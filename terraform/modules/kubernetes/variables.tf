variable "eks_enabled" {
  type    = bool
  default = true
}

variable "eks_cluster_name" {
  type = string
}

variable "aws_access_key_id" {
  type = string
}

variable "aws_access_key_secret" {
  type      = string
  sensitive = true
}
