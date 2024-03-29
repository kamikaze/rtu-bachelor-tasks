module "ssh_key_pair" {
  source = "terraform-aws-modules/key-pair/aws"

  key_name   = var.admin_ssh_public_key_name
  public_key = var.admin_ssh_public_key
}

resource "tls_private_key" "flux" {
  algorithm   = "ECDSA"
  ecdsa_curve = "P256"
}
